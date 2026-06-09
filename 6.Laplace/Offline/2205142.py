import numpy as np
import matplotlib.pyplot as plt

class SmartIrrigation:
    def __init__(self, a=0.5, b=1.0, t_max=20, dt=0.01):
        self.a=a
        self.b=b
        self.t_max=t_max
        self.dt=dt
        self.t=np.arange(0,self.t_max,self.dt)

    def u_step(self):
        return np.ones_like(self.t)
    def u_ramp(self):
        return self.t*0.1
    def u_sin(self):
        return np.sin(0.5*self.t)
    def u_exponential(self): 
        return 1- np.exp(-0.3*self.t)
    def u_pulse(self):
        u=np.zeros_like(self.t)
        u[self.t<5]=1
        return u

    def laplace_transform(self, f, s):
        exp_term=np.exp(-s*self.t)
        return np.trapezoid(f*exp_term,self.t)

    def inverse_laplace(self, s_list, F_s_values):
        W=np.max(np.imag(s_list))
        N=len(s_list)
        d_omega=2*W/N
        h_vals=[]
        for t in self.t:
            exp_term= np.exp(s_list*t)
            summ=np.sum(F_s_values*exp_term)
            h_t=(d_omega/(2*np.pi))*np.real(summ)
            h_vals.append(h_t)
        return np.array(h_vals)  #h(t)
    

    def H_s(self, s, U_s):
        return (self.b/(s+self.a))*U_s
    
    def steady_state(self, h):
        """Mean of last 5% of signal."""
        return np.mean(h[int(0.95 * len(h)) :])

    def time_constant(self, h):
        """Time to first reach 63.2% of steady-state."""
        h_ss=self.steady_state(h)
        idx=np.where(h>=0.632*h_ss)[0]
        return self.t[idx[0]] if len(idx) else np.nan

    def rise_time(self, h):
        """Time to go from 10% to 90% of steady-state."""
        h_ss = self.steady_state(h)
        idx_10 = np.where(h >= 0.1 * h_ss)[0]
        idx_90 = np.where(h >= 0.9 * h_ss)[0]
        if len(idx_10) and len(idx_90):
            return self.t[idx_90[0]] - self.t[idx_10[0]]
        return np.nan

    def settling_time(self, h):
        """Time after which h(t) stays permanently within ±2% of h_ss."""
        h_ss = self.steady_state(h)
        lower, upper = 0.98 * h_ss, 1.02 * h_ss
        for i in range(len(h) - 1, -1, -1):
            if h[i] < lower or h[i] > upper:
                return self.t[i + 1] if i + 1 < len(self.t) else self.t[-1]
        return self.t[0]

    def overshoot(self, h):
        """Percentage overshoot: (h_max - h_ss) / h_ss * 100."""
        h_ss = self.steady_state(h)
        h_max = np.max(h)
        if h_max <= h_ss:
            return 0.0
        return ((h_max - h_ss) / h_ss) * 100

    def compute_metrics(self, h):
       
        return {
            "steady_state":  self.steady_state(h),
            "time_constant": self.time_constant(h),
            "rise_time":     self.rise_time(h),
            "settling_time": self.settling_time(h),
            "overshoot_%":   self.overshoot(h),
        }

    def euler_simulate(self, u):
        """
        Euler method for dh/dt = -a*h(t) + b*u(t)
        h[n+1] = h[n] + dt * (-a*h[n] + b*u[n])
        """
        h = np.zeros_like(self.t)
        for n in range(len(self.t) - 1):
            dhdt = -self.a * h[n] + self.b * u[n]
            h[n + 1] = h[n] + self.dt * dhdt
        return h


#Change values of a, b to experiment with different system dynamics

a_vals=[0.5,1,1.5]
b_vals=[1.2,1.5,2]

for it in range(0,3,1):
    print("a : ",a_vals[it])
    print("b : ",b_vals[it])
    


    system = SmartIrrigation(a=a_vals[it], b=b_vals[it], t_max=20, dt=0.01)

    inputs = {
        "Step Input":        system.u_step(),
        "Ramp Input":        system.u_ramp(),
        "Sinusoidal Input":  system.u_sin(),
        "Exponential Input": system.u_exponential(),
        "Pulse Input":       system.u_pulse(),
    }

    # Bromwich contour parameters, set these values
    c = 0.4
    W = 50.0
    N = 1000
    w=np.linspace(-W,W,N)
    s_list = c+ 1j *w

    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

    for idx, (name, u) in enumerate(inputs.items()):
        print(f"Processing: {name}...")

        # --- Laplace --- set these values
        U_s_vals = np.array([system.laplace_transform(u, s) for s in s_list])
        H_s_vals = system.H_s(s_list, U_s_vals)
        h_laplace = system.inverse_laplace(s_list, H_s_vals)
        print(f"\n  ► {name}")
        metrics = system.compute_metrics(h_laplace)
        for k, v in metrics.items():
            print(f"      {k.replace('_',' ').title():<22}: {v}")

        # --- Euler ---
        h_euler = system.euler_simulate(u)

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        fig.suptitle(f"Smart Irrigation — {name}", fontsize=13, fontweight='bold')

        # Laplace subplot
        axes[0].plot(system.t, u, 'b--', lw=1.8, label="Input u(t)")
        axes[0].plot(system.t, h_laplace, color=colors[idx], lw=2.2, label="Output h(t)")
        axes[0].set_title("Laplace Transform Simulation", fontweight='bold')
        axes[0].set_xlabel("Time (s)", fontsize=11)
        axes[0].set_ylabel("Water Level / Input", fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Euler subplot
        axes[1].plot(system.t, u, 'b--', lw=1.8, label="Input u(t)")
        axes[1].plot(system.t, h_euler, color='tomato', lw=2.2, label="Output h(t)")
        axes[1].set_title("Euler Method Simulation", fontweight='bold')
        axes[1].set_xlabel("Time (s)", fontsize=11)
        axes[1].set_ylabel("Water Level / Input", fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()