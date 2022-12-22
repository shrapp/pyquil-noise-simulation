# pyquil-noise-simulation

# To Emanuelle

Do we add noise as I or on every specefic gate? (Is there any physical differene?)

Why T2 is upper bounded by 2 * T1 ?

Which way to build damping and dephasing kraus operators is better? (3K vs 4K)

Which gates should have noise?

# To Rigetti

Does gates hapeen in parallel?

How much time each gate takes?

Why the function get_noisy_gate(g.name, g.params) exist?

# To us

When does the current QC add the noise model to program?

Build new implementation for:
1. apply_noise_model
2. _decoherence_noise_model

without get_noisy_gate