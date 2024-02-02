import csv

class ReduceToSingleCaption:
    def reduce_to_single_caption(input_file, output_file):
        # Dictionary to store unique image names and corresponding captions
        unique_captions = {}

        with open(input_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip the header row

            for row in reader:
                image_name, caption = row[0], row[1]
                
                # If the image name is not in the dictionary, add it with the current caption
                if image_name not in unique_captions:
                    unique_captions[image_name] = caption

        # Write the unique image names and captions to a new CSV file
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # Write the header

            for image_name, caption in unique_captions.items():
                writer.writerow([image_name, caption])

