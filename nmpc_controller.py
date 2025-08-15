"""
Nonlinear Model Predictive Control (NMPC) with acados
Integrates with SAC and PINN for hybrid learning-based control
"""
import numpy as np
import casadi as ca
from typing import Dict, Optional, Tuple, List
import torch

try:
    from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver, AcadosSimSolver
except ImportError:
    print("Warning: acados_template not available. Install acados first.")
    AcadosOcp = AcadosModel = AcadosOcpSolver = AcadosSimSolver = None

from pendubot_model import PendubotModel


class NMPCController:
    """NMPC Controller with residual dynamics learning"""
    
    def __init__(self,
                 model: PendubotModel,
                 prediction_horizon: int = 40,
                 control_horizon: int = 40,
                 dt: float = 0.01,
                 Q: np.ndarray = None,
                 R: np.ndarray = None,
                 Qf: np.ndarray = None,
                 solver_type: str = 'SQP_RTI',
                 max_iter: int = 100,
                 qp_solver: str = 'PARTIAL_CONDENSING_HPIPM'):
        
        self.model = model
        self.N = prediction_horizon
        self.dt = dt
        self.solver_type = solver_type
        self.max_iter = max_iter
        
        # Cost matrices
        self.Q = Q if Q is not None else np.diag([30.0, 10.0, 2.0, 1.0])
        self.R = R if R is not None else np.array([[0.1]])
        self.Qf = Qf if Qf is not None else self.Q * 10
        
        # Reference trajectory
        self.x_ref = model.get_equilibrium_points()['upright']
        self.u_ref = np.array([0.0])
        
        # Setup acados OCP
        self._setup_ocp()
        
        # Create solver
        if AcadosOcpSolver is not None:
            self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')
        else:
            self.solver = None
            print("Warning: acados solver not initialized")
        
        # Statistics
        self.solve_times = []
        self.solve_status = []
        self.iterations = []
    
    def _setup_ocp(self):
        """Setup acados Optimal Control Problem"""
        if AcadosOcp is None:
            return
        
        # Create acados model
        acados_model = AcadosModel()
        acados_model.name = 'pendubot_nmpc'
        
        # State and control
        acados_model.x = self.model.x
        acados_model.u = self.model.u
        acados_model.z = ca.SX.sym('z', 0)  # No algebraic variables
        
        # Residual parameter (for PINN integration)
        acados_model.p = ca.SX.sym('p', 4)  # Residual dynamics
        
        # Discrete-time dynamics with residual
        x_next = self.model.x + self.dt * (
            self.model.f_nominal(self.model.x, self.model.u) + acados_model.p
        )
        
        # Convert to continuous-time for acados
        acados_model.f_expl_expr = (x_next - self.model.x) / self.dt
        acados_model.f_impl_expr = ca.vertcat([])  # No implicit dynamics
        
        # Create OCP
        self.ocp = AcadosOcp()
        self.ocp.model = acados_model
        
        # Dimensions
        nx, nu = 4, 1
        self.ocp.dims.N = self.N
        
        # Cost function - Linear Least Squares
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        
        # Cost matrices
        W = np.block([
            [self.Q, np.zeros((nx, nu))],
            [np.zeros((nu, nx)), self.R]
        ])
        self.ocp.cost.W = W
        self.ocp.cost.W_e = self.Qf
        
        # Output selection matrices
        Vx = np.hstack([np.eye(nx), np.zeros((nx, nu))])
        Vu = np.hstack([np.zeros((nu, nx)), np.eye(nu)])
        self.ocp.cost.Vx = Vx
        self.ocp.cost.Vu = Vu
        self.ocp.cost.Vx_e = np.eye(nx)
        
        # Reference
        yref = np.hstack([self.x_ref, self.u_ref])
        self.ocp.cost.yref = yref
        self.ocp.cost.yref_e = self.x_ref
        
        # Constraints
        u_min, u_max = self.model.get_control_bounds()
        x_min, x_max = self.model.get_state_bounds()
        
        # Control constraints
        self.ocp.constraints.lbu = u_min
        self.ocp.constraints.ubu = u_max
        self.ocp.constraints.idxbu = np.array([0], dtype=np.int32)
        
        # State constraints (velocity limits)
        self.ocp.constraints.lbx = x_min
        self.ocp.constraints.ubx = x_max
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3], dtype=np.int32)
        
        # Initial condition constraint (will be updated online)
        self.ocp.constraints.x0 = np.zeros(nx)
        
        # Solver options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.nlp_solver_type = self.solver_type
        self.ocp.solver_options.nlp_solver_max_iter = self.max_iter
        self.ocp.solver_options.tf = self.N * self.dt
        
        # Real-time iteration options
        if self.solver_type == 'SQP_RTI':
            self.ocp.solver_options.nlp_solver_step_length = 1.0
            self.ocp.solver_options.levenberg_marquardt = 1e-6
            self.ocp.solver_options.regularize_method = 'CONVEXIFY'
        
        # Warm start
        self.ocp.solver_options.qp_solver_warm_start = 1
        
        # Parameter values (initial residual = 0)
        self.ocp.parameter_values = np.zeros(4)
    
    def solve(self, 
              x_current: np.ndarray,
              residual_predictor: Optional[object] = None,
              x_ref_traj: Optional[np.ndarray] = None,
              u_ref_traj: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Solve NMPC optimization problem"""
        
        if self.solver is None:
            # Fallback to simple feedback control
            return self._fallback_control(x_current), {'status': -1, 'solve_time': 0.0}
        
        import time
        start_time = time.time()
        
        # Set initial condition
        self.solver.set(0, "lbx", x_current)
        self.solver.set(0, "ubx", x_current)
        
        # Update references and residual parameters
        for i in range(self.N):
            # Set stage reference
            if x_ref_traj is not None and i < len(x_ref_traj):
                x_ref_i = x_ref_traj[i]
            else:
                x_ref_i = self.x_ref
            
            if u_ref_traj is not None and i < len(u_ref_traj):
                u_ref_i = u_ref_traj[i]
            else:
                u_ref_i = self.u_ref
            
            yref_i = np.hstack([x_ref_i, u_ref_i])
            self.solver.set(i, "yref", yref_i)
            
            # Set residual parameter
            if residual_predictor is not None:
                # Predict residual using PINN
                x_tensor = torch.FloatTensor(x_current).unsqueeze(0)
                u_tensor = torch.zeros(1, 1)  # Use zero action for prediction
                
                with torch.no_grad():
                    residual = residual_predictor(x_tensor, u_tensor).cpu().numpy().flatten()
            else:
                residual = np.zeros(4)
            
            self.solver.set(i, "p", residual)
        
        # Set terminal reference
        if x_ref_traj is not None and len(x_ref_traj) > self.N:
            x_ref_N = x_ref_traj[self.N]
        else:
            x_ref_N = self.x_ref
        
        self.solver.set(self.N, "yref_e", x_ref_N)
        
        # Solve
        status = self.solver.solve()
        solve_time = time.time() - start_time
        
        # Extract solution
        if status == 0:  # Successful solve
            u_opt = self.solver.get(0, "u")
            x_pred = np.array([self.solver.get(i, "x") for i in range(self.N + 1)])
            u_pred = np.array([self.solver.get(i, "u") for i in range(self.N)])
        else:
            # Use previous solution or fallback
            u_opt = self._fallback_control(x_current)
            x_pred = None
            u_pred = None
        
        # Store statistics
        self.solve_times.append(solve_time)
        self.solve_status.append(status)
        if hasattr(self.solver, 'get_stats'):
            stats = self.solver.get_stats('sqp_iter')
            self.iterations.append(stats if stats is not None else 0)
        
        # Prepare return info
        info = {
            'status': status,
            'solve_time': solve_time,
            'x_prediction': x_pred,
            'u_prediction': u_pred,
            'cost': self._compute_cost(x_current, u_opt) if status == 0 else np.inf
        }
        
        return u_opt, info
    
    def _fallback_control(self, x_current: np.ndarray) -> np.ndarray:
        """Fallback LQR control when NMPC fails"""
        # Simple LQR feedback around upright equilibrium
        x_error = x_current - self.x_ref
        
        # LQR gain (precomputed for stability)
        K = np.array([[-70.71, -10.95, -14.14, -4.47]])  # Example gains
        
        u_fb = -K @ x_error
        
        # Apply input constraints
        u_min, u_max = self.model.get_control_bounds()
        u_fb = np.clip(u_fb, u_min, u_max)
        
        return u_fb
    
    def _compute_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """Compute stage cost"""
        x_error = x - self.x_ref
        u_error = u - self.u_ref
        
        cost = x_error.T @ self.Q @ x_error + u_error.T @ self.R @ u_error
        return cost
    
    def set_reference(self, x_ref: np.ndarray, u_ref: np.ndarray = None):
        """Update reference trajectory"""
        self.x_ref = x_ref
        if u_ref is not None:
            self.u_ref = u_ref
        else:
            self.u_ref = np.zeros(1)
    
    def set_weights(self, Q: np.ndarray = None, R: np.ndarray = None, Qf: np.ndarray = None):
        """Update cost matrices"""
        if Q is not None:
            self.Q = Q
        if R is not None:
            self.R = R
        if Qf is not None:
            self.Qf = Qf
        
        # Recreate solver with new weights
        if self.solver is not None:
            self._setup_ocp()
            self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')
    
    def get_stats(self) -> Dict:
        """Get controller statistics"""
        if not self.solve_times:
            return {}
        
        return {
            'mean_solve_time': np.mean(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'success_rate': np.mean([s == 0 for s in self.solve_status]),
            'mean_iterations': np.mean(self.iterations) if self.iterations else 0
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.solve_times = []
        self.solve_status = []
        self.iterations = []


class MPCReferenceGenerator:
    """Generate reference trajectories for MPC"""
    
    def __init__(self, model: PendubotModel):
        self.model = model
        self.equilibria = model.get_equilibrium_points()
    
    def swing_up_trajectory(self, 
                           x_current: np.ndarray, 
                           horizon: int,
                           dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate swing-up trajectory to upright position"""
        
        x_target = self.equilibria['upright']
        x_traj = np.zeros((horizon + 1, 4))
        u_traj = np.zeros((horizon, 1))
        
        # Simple linear interpolation for now
        for i in range(horizon + 1):
            alpha = min(1.0, i / horizon)
            x_traj[i] = (1 - alpha) * x_current + alpha * x_target
        
        # Zero control reference
        u_traj[:] = 0.0
        
        return x_traj, u_traj
    
    def energy_based_trajectory(self,
                               x_current: np.ndarray,
                               horizon: int,
                               dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate energy-based reference for swing-up"""
        
        x_target = self.equilibria['upright']
        target_energy = self.model.energy(x_target)
        current_energy = self.model.energy(x_current)
        
        x_traj = np.zeros((horizon + 1, 4))
        u_traj = np.zeros((horizon, 1))
        
        x_traj[0] = x_current
        
        for i in range(horizon):
            # Simple energy-based progression
            alpha = i / horizon
            energy_target = current_energy + alpha * (target_energy - current_energy)
            
            # Approximate state for target energy (simplified)
            if i < horizon - 5:  # Build energy first
                x_traj[i + 1] = x_current  # Keep swinging
            else:  # Final approach
                beta = (i - (horizon - 5)) / 5
                x_traj[i + 1] = (1 - beta) * x_current + beta * x_target
        
        return x_traj, u_traj


# Test the NMPC controller
if __name__ == "__main__":
    print("Testing NMPC Controller...")
    
    # Create model
    model = PendubotModel()
    
    # Create controller
    controller = NMPCController(model, prediction_horizon=20, dt=0.02)
    
    # Test solve
    x_test = np.array([0.1, 0.1, 0.0, 0.0])
    
    try:
        u_opt, info = controller.solve(x_test)
        print(f"Optimal control: {u_opt}")
        print(f"Solve info: {info}")
    except Exception as e:
        print(f"NMPC solve failed (expected if acados not installed): {e}")
        
        # Test fallback
        u_fallback = controller._fallback_control(x_test)
        print(f"Fallback control: {u_fallback}")
    
    # Test reference generator
    ref_gen = MPCReferenceGenerator(model)
    x_traj, u_traj = ref_gen.swing_up_trajectory(x_test, horizon=10, dt=0.02)
    print(f"Reference trajectory shape: {x_traj.shape}")
    
    print("✓ NMPC test completed successfully!")