import re
import concurrent.futures

def reformat_description(line):
    if not line.strip():
        return ''  

    pattern = r'"(.*?)"""",(.*?),\s*(Bug|Story|Task|Epic|Improvement|New Feature|Sub-task|Suggestion)'

    try:
        match = re.search(pattern, line)
        if match:

            description = match.group(2).replace(',', ';')
            formatted_line = line[:match.start(2)] + description + line[match.end(2):]
            return formatted_line
        return line  
    except Exception as e:
        print(f"Error: {e} | Line: {line.strip()[:100]}...")
        return '' 
def reformat_and_clean_csv_streamed(input_csv, temp_csv, timeout_sec=5):
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(temp_csv, 'w', encoding='utf-8') as outfile, \
         concurrent.futures.ProcessPoolExecutor() as executor:

        first_line = infile.readline()
        first_line = first_line.replace('Description_Text,', '').replace('Description_Code,', '')
        outfile.write(first_line)

        for idx, line in enumerate(infile, 1):
            future = executor.submit(reformat_description, line)
            try:
                formatted = future.result(timeout=timeout_sec)
                if formatted:
                    outfile.write(formatted)
            except concurrent.futures.TimeoutError:
                print(f"[Timeout] Line {idx} took too long. Skipping.")
            except Exception as e:
                print(f"[Error] Line {idx}: {e}")

if __name__ == '__main__':
    input_file = 'full/issues_43825.csv'
    temp_csv = 'temp/issues_43825.csv'

    reformat_and_clean_csv_streamed(input_file, temp_csv)
