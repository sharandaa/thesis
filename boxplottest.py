import matplotlib.pyplot as plt

# Sample data
#data = {'ESPCN': [1, 2, 3, 4, 15], 'SwinIR': [2, 3, 14, 5, 6], 'RealESRGAN': [3, 4, -5, 6, 7]}

# Create a list of lists containing the data for each group
group_data = [values for key, values in data.items()]

# Create a box plot
fig, ax = plt.subplots()
ax.boxplot(group_data)

# Add labels and title
ax.set_xticklabels(data.keys())
#ax.set_xlabel('Groups')
ax.set_ylabel('PSNR (dB)')
ax.set_title('PSNR scores of upsampling with scale x2')

# Display the plot
plt.show()
