import numpy as np
import matplotlib.pyplot as plt
import os

class ContinuousSignal:
    def __init__(self, func):
        self.func = func


    def shift(self, t_shift):
        def shifted_func(t):
            return self.func(t - t_shift)
        return ContinuousSignal(shifted_func)


    def add(self, other):
        def added_func(t):
            return self.func(t) + other.func(t)
        return ContinuousSignal(added_func)


    def multiply(self, other):
        def multiplied_func(t):
            return self.func(t) * other.func(t)
        return ContinuousSignal(multiplied_func)


    def multiply_const_factor(self, scalar):
        def scaled_func(t):
            return scalar * self.func(t)
        return ContinuousSignal(scaled_func)


    def plot(self, t_min, t_max, num_points=500, title="Continuous Signal", ax=None):
        t = np.linspace(t_min, t_max, num_points)
        y = self.func(t)
        if ax is None:
            plt.figure(figsize=(8, 4))
            plt.plot(t, y)
            plt.title(title)
            plt.xlabel("t (Time)")
            plt.ylabel("x(t)")
            plt.grid(True)
        else:
            ax.plot(t, y)
            ax.set_title(title)
            ax.grid(True)


class LTI_Continuous:
    def __init__(self, impulse_response):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal, delta, t_range):
        
        impulses = []
        coeffs = []
        
        t_min, t_max = t_range
        
        tk_values = np.arange(t_min, t_max, delta)
        
        for tk in tk_values:
            
            ck = input_signal.func(tk) * delta
            
            # Define shifted rectangular impulse delta_delta(t - tk) [cite: 35]
            def rectangular_pulse(t, tk_fixed=tk):
                time_diff = t - tk_fixed
                # Height is 1/delta for 0 <= t < delta [cite: 34]
                return np.where((time_diff >= 0) & (time_diff < delta), 1.0/delta, 0.0)
            
            impulses.append(ContinuousSignal(rectangular_pulse))
            coeffs.append(ck)
            
        return impulses, coeffs

    def output_approx(self, input_signal, delta, t_range):
        
        # Get the decomposition coefficients c_k and timing from the input [cite: 30]
        _, coeffs = self.linear_combination_of_impulses(input_signal, delta, t_range)
        
        t_min, t_max = t_range
        tk_values = np.arange(t_min, t_max, delta)

        def result_func(t):
            total_output = np.zeros_like(t, dtype=float)
            for ck, tk in zip(coeffs, tk_values):
                # We shift the system's h(t) by tk and scale by ck
                # h_shifted(t) = h(t - tk)
                total_output += ck * self.impulse_response.func(t - tk)
            return total_output

        return ContinuousSignal(result_func)


def main():
    # Define unit step u(t)
    def unit_step(t):
        return np.where(t >= 0, 1.0, 0.0)
    
    # Define x(t) = e^-t * u(t) [cite: 44]
    def input_func(t):
        return np.exp(-t) * unit_step(t)
    
    # Initialize signals and system [cite: 42]
    x = ContinuousSignal(input_func)
    h = ContinuousSignal(unit_step) # h(t) = u(t) [cite: 45]
    system = LTI_Continuous(h)
    
    T = 3
    delta = 0.05
    
    # Calculate approximation [cite: 31]
    y_hat = system.output_approx(x, delta, (-T, T))
    
    # Plotting
    plt.figure(figsize=(10, 5))
    t_plot = np.linspace(-T, T*2, 2000)
    plt.plot(t_plot, y_hat.func(t_plot), label=f"Approx Output (delta={delta})")
    plt.title("Approximate System Output $y(t)$")
    plt.xlabel("t")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()




# def main():
    
#     output_dir = "continuous_practice"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

    
#     def unit_step(t):
#         return np.where(t >= 0, 1.0, 0.0)
    
    
#     def input_func(t):
#         return np.exp(-t) * unit_step(t)
    
#     T = 3
#     x = ContinuousSignal(input_func)
#     h = ContinuousSignal(unit_step)
#     system = LTI_Continuous(h)

#     # Figure 1: Input Signal [cite: 56]
#     plt.figure(figsize=(8, 4))
#     x.plot(-T, T, title=f"Figure 1: Input Signal x(t), INF={T}")
#     plt.savefig(f"{output_dir}/figure1_input.png")
#     plt.close()

#     # Figure 2: Reconstruction and Grid [cite: 73, 74]
#     delta_val = 0.5
#     imps, cks = system.linear_combination_of_impulses(x, delta_val, (-T, T))
    
#     def reconstructed_func(t):
#         # x_hat(t) = sum of c_k * impulse_k(t) [cite: 52, 53]
#         total = 0
#         for c, imp in zip(cks, imps):
#             total += c * imp.func(t)
#         return total

#     # Plot grid (truncated for brevity, same logic as discrete plot)
#     fig, axes = plt.subplots(4, 3, figsize=(15, 12))
#     axes = axes.flatten()
#     for i in range(min(11, len(cks))):
#         # Component x_k(t) = c_k * delta_delta(t - t_k) [cite: 74]
#         def comp_func(t, coeff=cks[i], pulse=imps[i]):
#             return coeff * pulse.func(t)
#         ContinuousSignal(comp_func).plot(-T, T, ax=axes[i], title=f"Component {i}")
    
#     ContinuousSignal(reconstructed_func).plot(-T, T, ax=axes[11], title="Reconstructed Signal [cite: 74]")
#     plt.savefig(f"{output_dir}/figure2_grid.png")
#     plt.close()

#     # Figure 3: Varying Delta [cite: 75, 77]
#     deltas = [0.5, 0.1, 0.05, 0.01]
#     fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
#     axes3 = axes3.flatten()

#     for i, d in enumerate(deltas):
#         cur_imps, cur_cks = system.linear_combination_of_impulses(x, d, (-T, T))
        
#         def rec_overlay(t, cs=cur_cks, ms=cur_imps):
#             res = 0
#             for c_val, m_val in zip(cs, ms):
#                 res += c_val * m_val.func(t)
#             return res
        
#         t_vals = np.linspace(-T, T, 500)
#         axes3[i].plot(t_vals, x.func(t_vals), label="x(t) [cite: 77]")
#         axes3[i].plot(t_vals, rec_overlay(t_vals), '--', label=f"Reconstructed ($\Delta={d}$)")
#         axes3[i].set_title(f"$\Delta = {d}$")
#         axes3[i].legend()

#     plt.savefig(f"{output_dir}/figure3_varying_delta.png")
#     plt.show()

# if __name__ == "__main__":
#     main()

