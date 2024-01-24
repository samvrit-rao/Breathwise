import pandas as pd
import zipfile
import os
import openai

p = pd.read_csv('/Users/raosamvr/Downloads/mimic-iii-clinical-database-1.4/PRESCRIPTIONS.csv')
i = pd.read_csv('/Users/raosamvr/Downloads/pca/second_iter.csv')

main = p[p['SUBJECT_ID'].isin(i['SUBJECT_ID'].tolist())]
openai.api_key = "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"


uno = i['SUBJECT_ID'].tolist()
dos = main['SUBJECT_ID'].unique().tolist()
def find_index_in_other_list(list1, list2, n):
    """
    Finds the index of the element at index n in list1, in list2.
    
    :param list1: The first list.
    :param list2: The second list.
    :param n: The index in list1 to check.
    :return: The index of the element in list2, or -1 if not found.
    """
    if n < len(list1):
        element = list1[n]
        return list2.index(element) if element in list2 else -1
    return -1

directory = "/Users/raosamvr/Downloads/pca/gibbs/" 
file_name = "output.txt"
filepath = os.path.join(directory, file_name)

# Your provided indices and the two lists to compare
indices = [34, 50, 53, 55, 56, 59, 76, 84, 92, 100, 147, 153, 160, 169, 183, 189, 204, 238, 239, 241, 267, 300, 310, 313, 328, 331, 336, 339, 357, 358, 365, 370, 379, 407, 433, 439, 458, 459, 479, 502, 503, 510, 517, 521, 558, 563, 604, 610]
# Open a file to write the results
with open(filepath, 'w') as file:
    for n in indices:
        index_in_list2 = find_index_in_other_list(uno, dos, n)
        file.write(f"Index {n} in list1 (element: '{uno[n] if n < len(uno) else 'Out of Range'}') is found at index {index_in_list2} in list2\n")

print("Results written to output.txt.")




def call_chatgpt_api(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are to read a list of drugs for patients with copd. you have 8 different cateogires:     Inhaled Corticosteroids: Used to reduce inflammation in the airways.Short-Acting Beta Agonists (SABAs): Provide quick relief from acute symptoms.Long-Acting Beta Agonists (LABAs): Help to keep the airways open for a longer duration.Anticholinergics: Reduce bronchospasm and help open airways.Phosphodiesterase-4 Inhibitors: Reduce inflammation and relax airways.Mucolytics: Help to thin mucus in the airways, making it easier to cough up.Oral Steroids: Used for short-term treatment during COPD exacerbations. Bronchodilators - relax muscles in the lungs. -> you are to read the list of drugs and classify them and see if they fall under that. if so, you are to write to a text file the category colon (:) either 1 or 0, depending on if they have it (1) or not (0). I do NOT WANT ANY BIG EXPLANATION. ALL YOU SHOULD BE OUTPUTTING TO THE TEXT FILE IS SOMETHING ALONG THESE LINES Inhaled Corticosteroids:1 Short-Acting Beta Agonists:1 Long-Acting Beta Agonists:0 Anticholinergics:1 Phosphodiesterase-4 Inhibitors:1 Mucolytics:0 Oral Steroids:0 Bronchidilators:0. Obviously, the thing I just said is an example and shouldn't be copied exactly, but you get what I mean; i just want the ones and zeros and no explanations. PLEASE DON'T GIVE ME ANY ADDITIONAL INFORMATION OR EXPLANATION - YOU DOING THIS HURTS MY TOKEN BALANCE MEANING I HAVE TO PAY MORE - YOUR JOB IS TO JUST GIVE ME THE CLASSIFCIATION LIKE I REQEUSTED - EVEN IF THINGS FALL UNDER MORE THAN ONE THING PLEASE JUST GIVE ME THE 0/1/0/1 CLASSIFICATION IN THE FORMAT I REQUESTED TO THE BEST OF YOUR ABILIITY."},
            {"role": "user", "content": text}
        ]
    )
    return response

def process_drug_list(drug_list):
    drugs_text = ', '.join(drug_list)
    return drugs_text

def process_subjects(df, subject_ids, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    counter = 0  

    for subject_id in subject_ids:
        feedback_file_path = os.path.join(output_path, f'feedback_subject_{counter}.txt')

        if os.path.exists(feedback_file_path):
            print(f"Skipping subject ID {subject_id} (file number {counter}) as feedback already exists.")
            counter += 1
            continue
        
        subject_df = df[df['SUBJECT_ID'] == subject_id]

        drug_list = subject_df['DRUG'].tolist()
        drug_set = set(drug_list)
        drug_list = list(drug_set)  
        drugs_text = process_drug_list(drug_list) 
        print(drugs_text)
        response = call_chatgpt_api(drugs_text)  
        feedback = response['choices'][0]['message']['content'].strip()

        # Write the feedback to a file
        with open(feedback_file_path, 'w') as output_file:
            output_file.write("Feedback:\n" + feedback)

        print(f"Processed subject ID {subject_id} and saved feedback to {feedback_file_path}")
        counter += 1  

subject_ids = main['SUBJECT_ID'].unique()
process_subjects(main, subject_ids, '/Users/raosamvr/Downloads/pca/gibbs/')
