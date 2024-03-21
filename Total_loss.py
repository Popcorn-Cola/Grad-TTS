import re


def extract_numbers(line):
    match = re.search(r'duration loss = (\d+) \| prior loss = (\d+) \| diffusion loss = (\d+)', line)
    if match:
        duration_loss = int(match.group(1))
        prior_loss = int(match.group(2))
        diffusion_loss = int(match.group(3)
        return duration_loss, prior_loss, diffusion_loss
    else:
        return None


        with open('train_backup.log', 'r') as infile, open('train1.log', 'w') as outfile:
        # Iterate through each line in the input file
        for line in infile:
            # Extract numbers from the line
            numbers = extract_numbers(line)
            if numbers is not None:
                duration_loss, prior_loss , diffusion_loss = numbers
                #Write the extracted numbers back to the output file
                outfile.write(f'duration loss = {duration_loss} | prior loss = {prior_loss} | diffusion loss ={diffusion_loss} | Total loss = {duration_loss + prior_loss + diffusion_loss}\n')

