import csv
import sys

csv.field_size_limit(10**7)

input_file = "issues.csv"
output_file = "full/issues_447133.csv"
start_line = 447133     
lines_to_read = 27000   

with open(input_file, "r", newline="", encoding="utf-8") as infile, open(
    output_file, "w", newline="", encoding="utf-8"
) as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    writer.writerow(header)

    for _ in range(start_line - 1):
        try:
            next(reader)
        except StopIteration:
            print("Reached end of file before hitting start_line.")
            break

    written = 0
    for row_num, row in enumerate(reader, start=start_line):
        if written >= lines_to_read:
            break
        try:
            writer.writerow(row)
            written += 1
        except csv.Error as e:
            print(f"Skipped row {row_num} due to CSV error: {e}")
            continue

print(f"Finished. Wrote {written} rows starting from line {start_line} to '{output_file}'.")
