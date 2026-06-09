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






system = SmartIrrigation(a=0.5, b=1.2, t_max=20, dt=0.01)


# PROBLEM 4 — Laplace Property Verifier


print("\n" + "="*60)
print("PROBLEM 4: LAPLACE PROPERTY VERIFICATION")
print("="*60)

t      = system.t
dt     = system.dt
t_max  = system.t_max

# Shared s-values for evaluation (real axis samples)
# We test on real s values so results are easy to compare
s_test = np.linspace(0.5, 5.0, 200)


# PROPERTY 1 — TIME SHIFTING
# L{x(t - t0)} == e^(-s*t0) * L{x(t)}
# We use x(t) = step input, t0 = 2


t0 = 2.0

# Left side: directly compute L{u(t - t0)}
u_original   = system.u_step()           # u(t)
u_shifted    = np.zeros_like(t)
u_shifted[t >= t0] = 1.0                # u(t - 2): step delayed by 2s

LHS_shift = np.array([system.laplace_transform(u_shifted, s) for s in s_test])

# Right side: e^(-s*t0) * L{u(t)}
U_original    = np.array([system.laplace_transform(u_original, s) for s in s_test])
RHS_shift     = np.exp(-s_test * t0) * U_original

# Error
error_shift   = np.abs(LHS_shift - RHS_shift)
print(f"\nProperty 1 — Time Shifting (t0 = {t0}s)")
print(f"  Max absolute error : {np.max(error_shift):.6f}")
print(f"  Mean absolute error: {np.mean(error_shift):.6f}")


# PROPERTY 2 — TIME DIFFERENTIATION
# L{dx/dt} == s * X(s)
# We use x(t) = exponential input


x        = system.u_exponential()        # x(t) = 1 - e^(-0.3t)

# Numerically differentiate x(t) using np.gradient
dxdt     = np.gradient(x, dt)           # dx/dt ≈ 0.3 * e^(-0.3t)

# Left side: directly compute L{dx/dt}
LHS_diff = np.array([system.laplace_transform(dxdt, s) for s in s_test])

# Right side: s * L{x(t)}
X_s      = np.array([system.laplace_transform(x, s) for s in s_test])
RHS_diff = s_test * X_s

# Error
error_diff = np.abs(LHS_diff - RHS_diff)
print(f"\nProperty 2 — Time Differentiation")
print(f"  Max absolute error : {np.max(error_diff):.6f}")
print(f"  Mean absolute error: {np.mean(error_diff):.6f}")


# PLOTS — Side by side for both properties


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Problem 4 — Laplace Property Verification", 
             fontsize=14, fontweight='bold')



# Left: time-domain signals
axes[0,0].plot(t, u_original, 'b-',  lw=2,   label="u(t) — original")
axes[0,0].plot(t, u_shifted,  'r--', lw=2,   label=f"u(t−{t0}) — shifted")
axes[0,0].axvline(t0, color='gray', linestyle=':', lw=1.5, label=f"t = {t0}s")
axes[0,0].set_title("Time Domain — Shifting", fontweight='bold')
axes[0,0].set_xlabel("Time (s)")
axes[0,0].set_ylabel("Amplitude")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Right: s-domain comparison
axes[0,1].plot(s_test, np.abs(LHS_shift), 'b-',  lw=2.2, label="|L{u(t−t0)}| — direct")
axes[0,1].plot(s_test, np.abs(RHS_shift), 'r--', lw=2.2, label="|e^{-st0}·U(s)| — property")
axes[0,1].plot(s_test, error_shift,       'k:',  lw=1.5, label="Absolute error")
axes[0,1].set_title("s-Domain — Time Shift Verification", fontweight='bold')
axes[0,1].set_xlabel("s (real)")
axes[0,1].set_ylabel("Magnitude")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

#Row 2: Time Differentiation

# Left: time-domain signals
axes[1,0].plot(t, x,    'b-',  lw=2, label="x(t) = 1 − e^{−0.3t}")
axes[1,0].plot(t, dxdt, 'r--', lw=2, label="dx/dt ≈ 0.3·e^{−0.3t}")
axes[1,0].set_title("Time Domain — Differentiation", fontweight='bold')
axes[1,0].set_xlabel("Time (s)")
axes[1,0].set_ylabel("Amplitude")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Right: s-domain comparison
axes[1,1].plot(s_test, np.abs(LHS_diff), 'b-',  lw=2.2, label="|L{dx/dt}| — direct")
axes[1,1].plot(s_test, np.abs(RHS_diff), 'r--', lw=2.2, label="|s·X(s)| — property")
axes[1,1].plot(s_test, error_diff,       'k:',  lw=1.5, label="Absolute error")
axes[1,1].set_title("s-Domain — Differentiation Verification", fontweight='bold')
axes[1,1].set_xlabel("s (real)")
axes[1,1].set_ylabel("Magnitude")
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()