import numpy as np

arr = np.load('./clipped_data/mdd_control.npy')

min_vals = np.min(arr, axis=(1, 2))
max_vals = np.max(arr, axis=(1, 2))

arr_normalized = (arr - min_vals[:, None, None]) / (max_vals - min_vals)[:, None, None] * 2 - 1

print(arr_normalized[0].max())
print(arr_normalized[0].min())

np.save(fr'clipped_data/mdd_control',arr_normalized )
# we're being lazy today boys
arr = np.load('./clipped_data/mdd_patient.npy')

min_vals = np.min(arr, axis=(1, 2))
max_vals = np.max(arr, axis=(1, 2))

arr_normalized = (arr - min_vals[:, None, None]) / (max_vals - min_vals)[:, None, None] * 2 - 1

print(arr_normalized[0].max())
print(arr_normalized[0].min())

np.save(fr'clipped_data/mdd_patient',arr_normalized )