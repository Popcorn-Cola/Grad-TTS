import re
import matplotlib.pyplot as plt


def extract_numbers(line):
    numbers = re.findall(r'\d+\.\d+', line)
    duration_loss = round(float(numbers[0]), 3)
    prior_loss = round(float(numbers[1]), 3)
    diffusion_loss = round(float(numbers[2]), 3)
    return duration_loss, prior_loss, diffusion_loss


with open('train.log', 'r') as infile, open('train1.log', 'w') as outfile:
# Iterate through each line in the input file
    duration_losses = []
    prior_losses = []
    diffusion_losses = []
    Total_losses = []
    for line in infile:
        # Extract numbers from the line
        numbers = extract_numbers(line)
        if numbers is not None:
            duration_loss, prior_loss , diffusion_loss = numbers
            Total_loss = duration_loss + prior_loss + diffusion_loss
            #Write the extracted numbers back to the output file
            outfile.write(f'duration loss = {duration_loss} | prior loss = {prior_loss} | diffusion loss ={diffusion_loss} | Total loss = {Total_loss}\n')
            duration_losses.append(duration_loss)
            prior_losses.append(prior_loss)
            diffusion_losses.append(diffusion_loss)
            Total_losses.append(Total_loss)
            
x_ax = [i+1 for i in range(len(duration_losses))]
plt.xscale('log')
plt.plot(x_ax, duration_losses, color='blue', label='duration loss')
plt.plot(x_ax, prior_losses, color='green', label='prior loss')
plt.plot(x_ax, diffusion_losses, color='red', label='diffusion loss')
plt.plot(x_ax, Total_losses, color='orange', label='Total loss')


plt.xlabel('Epoch')
plt.title('Validation_loss')

plt.legend()

plt.savefig('Validation_loss.png')
    
