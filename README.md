Summary
-------

Title: Realistic Noise Simulation of Rigetti's QPU in PyQuil

This proposal aims to enhance the PyQuil simulator by introducing a more realistic noise simulation of Rigetti's QPU. The current implementation only supports basic noise models and doesn't accurately represent noise in Rigetti's QPU. The proposed solution includes features such as:

- Calibrations (Class) for handling recent calibration data
- Improved single qubit noise simulation (amplitude damping & dephasing)
- Depolarizing channel
- Readout noise
- Custom intensity parameter for noise model customization
- Support for non-native gates

These enhancements will enable researchers and developers to simulate quantum algorithms with realistic noise, allowing them to design error-correcting techniques tailored to Rigetti's platform.

Integration
------------

To integrate the new features, the code in new_noise.py should be copied into noise.py. This does not replace the existing module but rather extends its functionality.
