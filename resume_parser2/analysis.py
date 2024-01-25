# *******************************************************************************************

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
from io import StringIO
import json
import spacy
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.pdfparser import PDFParser
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfdocument import PDFDocument

file_path = 'resume_parser2/categorized_companies.csv'
df_companies = pd.read_csv(file_path)
nlp = spacy.load('en_core_web_sm')



#*****************************
def extracted_raw_data(text):
    output_string = StringIO()
    with open(text, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    pdf_content = output_string.getvalue().lower()
    return pdf_content

#*****************************


def visualize_analysis(analysis_results):
    labels = list(analysis_results.keys())
    values = list(analysis_results.values())

    fig, ax = plt.subplots()

    # Plot overall score as a horizontal bar
    overall_score = values[-1]
    ax.barh('Overall Score', overall_score, color='blue')

    # Plot individual scores as horizontal bars
    individual_scores = values[:-1]
    categories = labels[:-1]
    ax.barh(categories, individual_scores, color=['orange', 'green', 'red'])

    # Add labels and title
    ax.set_xlabel('Percentage')
    ax.set_title('Job Matching Analysis')

    # Display overall percentage on top of the bar
    plt.text(overall_score + 1, 0, f'{overall_score:.2f}%', va='center', fontsize=10)

    plt.show()


def analyze_job_match(applicant_data, job_description,total_score):
    # Mock data for job description
    job_skills = set(['Python', 'Django', 'Linux', 'Programming', 'C'])
    job_experience_required = 2  # Minimum required experience in years
    job_location = 'Mumbai'

    # Calculate skill match percentage
    applicant_skills = set(applicant_data['skills'])
    skill_match_percentage = (len(applicant_skills.intersection(job_skills)) / len(job_skills)) * 100

    # Check if experience matches
    experience_match = 1 if applicant_data['total_experience'] >= job_experience_required else 0

    # Check if location matches
    location_match = location_match(applicant_data['current_city'],job_location)

    # Calculate overall score
    overall_score = calculate_overall_score(skill_match_percentage, experience_match, location_match)

    # Visualize the analysis
    analysis_results = {
        'Skills Match Percentage': skill_match_percentage,
        'Experience Match': ((total_score / 36000) * 100),
        'Location Match': location_match,
        'Overall Score': overall_score
    }

    visualize_analysis(analysis_results)

    return analysis_results


def extract_skills(data, skills_file=None, job_skills=[]):
    texter = ' '.join(data.split())
    nlp_text = nlp(texter)
    noun_chunks = list(nlp_text.noun_chunks)
    tokens = [token.text for token in nlp_text if not token.is_stop]

    if not skills_file:
        data = pd.read_csv('resume_parser2/skills.csv')
    else:
        data = pd.read_csv(skills_file)

    skills = list(data.columns.values)
    skillset = []

    # check for one-grams
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    

    # Extracted skills from the dataset
    extracted_skills = [i.capitalize() for i in set([i.lower() for i in skillset])]
    # Add skills from job_skills that are not already in extracted_skills


    for skill in job_skills:
        if skill.lower() not in [s.lower() for s in extracted_skills]:
            extracted_skills.append(skill.capitalize())
            # print(f"Job skill not found in skills.csv: {skill}")

    # Split skills like 'DevOps and CI/CD' into separate skills 'DevOps' and 'CI/CD'
    final_skills = []
    for skill in extracted_skills:
        if 'and' in skill.lower():
            split_skills = [part.strip() for part in skill.split('and')]
            final_skills.extend(split_skills)
        else:
            final_skills.append(skill)

    # Print skills from job_skills that are not in the dataset (need to update this later)
    # for skill in final_skills:
    #     if skill.lower() not in [s.lower() for s in skills]:
    #         print(f"Skill from job_skills not in dataset: {skill}")

    return final_skills

def calculate_overall_score(skill_match_percentage, experience_match, location_match, similarity_score):
    # Adjust weights based on the importance of each factor
    skill_weight = 0.3
    experience_weight = 0.15
    location_weight = 0.1
    similarity_weight = 0.45

    # Calculate overall score
    overall_score = (skill_match_percentage * skill_weight +
                     experience_match * experience_weight +
                     location_match * location_weight +
                     similarity_score * similarity_weight)

    return overall_score



def location_matcher(applicant_location, job_location):
    score = 0
    file_path = 'resume_parser2/worldcities.csv'
    df = pd.read_csv(file_path, encoding='latin-1')
    dataset = df[['city', 'country', 'iso2', 'tier']]

    # Convert both applicant and job locations to lowercase and remove leading/trailing whitespaces
    applicant_location = applicant_location.strip().lower()
    job_location = job_location.strip().lower()

    if(applicant_location == job_location):
      score = 100
      return score
    else:
      # Check if there's an exact match in the dataset
      match_row = dataset[(dataset['city'].str.lower() == applicant_location) & (dataset['country'].str.lower() == 'india')]

      if not match_row.empty:
          # Retrieve tier values for both locations
          applicant_tier = match_row['tier'].iloc[0]
          job_tier = dataset.loc[dataset['city'].str.lower() == job_location, 'tier'].iloc[0]

          # Define the scoring rules
          scoring_rules = {
              (1, 1): 50, (2, 1): 35, (3, 1): 25, (4, 1): 0,
              (1, 2): 75, (2, 2): 50, (3, 2): 25, (4, 2): 0,
              (1, 3): 75, (2, 3): 65, (3, 3): 50, (4, 3): 25,
              (1, 4): 75, (2, 4): 65, (3, 4): 60, (4, 4): 50,
          }

          # Calculate the score based on the tier values
          score = scoring_rules.get((applicant_tier, job_tier), 0)

          print(f"Applicant Location: {applicant_location}, Tier: {applicant_tier}")
          print(f"Job Location: {job_location}, Tier: {job_tier}")
          print(f"Score: {score}")

          return score
      else:
          print("No exact match found in the dataset.")
          return 0

# Lists to store unknown matches
unknown_matches = []


job_experience_range = (2, 5)  # Minimum and maximum years of experience required
comp_vertical = 10

def calculate_experience_range_score(min_experience, max_experience,comp_vertical):

    industry, category = get_industry_for_company(comp_vertical)
    score = 6
    if category == 'A':
            score = 6
    elif category == 'B':
            score =  7
    elif category == 'C':
            score = 8
    elif category == 'D':
            score = 9
    elif category == 'E':
            score = 10
    else:
            score = 6

    min_score = min_experience * score * 12 * 10
    max_score = max_experience * score * 12 * 10
    return min_score, max_score


# Function to calculate duration in months
from datetime import datetime

def calculate_duration_in_months(date_range):
    print(f"Date Range: {date_range}")  # Add this line
    date_range = date_range.replace('’', "'")

    try:
        start_date_str, end_date_str = re.split(' – | - | To | to ', date_range)
    except ValueError:
        # If the split with full month name fails, try splitting with abbreviated month name
        try:
            start_date_str, end_date_str = re.split(' – | - ', date_range)
        except ValueError:
            # If only one date is provided, consider it as the start date and use "present" as the end date
            start_date_str = date_range.strip()
            end_date_str = 'present'

    def convert_apostrophe_format(date_str):
        # Add more date formats if needed
        formats = ["%d-%b-%Y", "%B %Y", "%b %Y", "%B %y", "%b %y"]
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime("%B %Y")
            except ValueError:
                pass
        raise ValueError("Unsupported date format: {}".format(date_str))
    
    if end_date_str.lower() == 'present' or end_date_str.lower() == 'till date':
        # Handle 'Present' or 'Till Date' case (use the current date as the end date)
        try:
            start_date_str = convert_apostrophe_format(start_date_str)
            start_date = datetime.strptime(start_date_str, "%B %Y")
        except ValueError:
            start_date_str = convert_apostrophe_format(start_date_str)
            try:
                start_date = datetime.strptime(start_date_str, "%b %Y")
            except ValueError:
                start_date = datetime.strptime(start_date_str, "%b %y")
        end_date = datetime.now()
    else:
        # Use "%B" for full month name and "%b" for abbreviated month name
        try:
            start_date_str = convert_apostrophe_format(start_date_str)
            end_date_str = convert_apostrophe_format(end_date_str)
            try:
                end_date = datetime.strptime(end_date_str, "%B %Y")
                start_date = datetime.strptime(start_date_str, "%B %Y")
            except ValueError:
                start_date = datetime.strptime(start_date_str, "%B %y")
                end_date = datetime.strptime(end_date_str, "%B %y")
        except ValueError:
            start_date_str = convert_apostrophe_format(start_date_str)
            end_date_str = convert_apostrophe_format(end_date_str)
            try:
                end_date = datetime.strptime(end_date_str, "%b %Y")
                start_date = datetime.strptime(start_date_str, "%b %Y")
            except ValueError:
                try:
                    start_date = datetime.strptime(start_date_str, "%B %y")
                    end_date = datetime.strptime(end_date_str, "%B %y")
                except ValueError:
                    start_date = datetime.strptime(start_date_str, "%b %y")
                    end_date = datetime.strptime(end_date_str, "%b %y")

    duration_in_months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
    return duration_in_months + 1

# Function to match company names and retrieve industry name and category
def get_industry_for_company(company_name):
    if company_name in df_companies['Organization Name'].values:
        matching_row = df_companies[df_companies['Organization Name'] == company_name]
        if not matching_row.empty:
            industry = matching_row['Industry'].iloc[0]
            category = matching_row['Category'].iloc[0]
            return industry, category
        else:
            return 'Unknown', 'Unknown'
    else:
        unknown_matches.append(company_name)
        return 'Unknown', 'Unknown'


def get_unique_words_from_sentences(sentences):
    unique_words = set()

    for sentence in sentences:
        words = sentence.split()
        unique_words.update(words)

    return list(unique_words)

def calculate_similarity(main_sentence, word_list):
    main_words = set(get_unique_words_from_sentences(main_sentence))
    list_words = set(word_list)

    common_words = main_words.intersection(list_words)
    print(f"com_len: {len(common_words)}")
    print(f"des_len: {len(list_words)}")
    print(f"main_len: {len(main_words)}")
    similarity_score = len(common_words) / len(main_words)

    return similarity_score

sentences =[]

def calculate_similarity_score(similarity_score):
    if similarity_score < 40:
        return similarity_score * 0.5
    elif 40 < similarity_score < 65:
        return similarity_score * 0.2
    else:
        return similarity_score



# Function to assign scores based on matching criteria and category
def assign_score(match_result, category):
    if match_result == 'Yes':
        if category == 'A':
            return 10 * 6
        elif category == 'B':
            return 10 * 7
        elif category == 'C':
            return 10 * 8
        elif category == 'D':
            return 10 * 9
        elif category == 'E':
            return 10 * 10
        else:
            return 100  # Default score
    elif match_result == 'No':
        if category == 'A':
            return 5 * 6
        elif category == 'B':
            return 5 * 7
        elif category == 'C':
            return 5 * 8
        elif category == 'D':
            return 5 * 9
        elif category == 'E':
            return 5 * 10
        else:
            return 50  # Default score
    else:
        if category == 'A':
            return 6 * 6
        elif category == 'B':
            return 6 * 7
        elif category == 'C':
            return 6 * 8
        elif category == 'D':
            return 6 * 9
        elif category == 'E':
            return 6 * 10
        else:
            return 60  # Default score

# Take industry name as input
target_industry = "Banking"
total_score = 0

def final_analysis(applicant_data,job_description,job_experience_range,comp_vertical,job_skills,comp_industry,job_location):
    total_score = 0
    total_duration = 0
    sim_s = 0

    for job_experience_key, job_experience_value in applicant_data['experience'].items():
        # Extract company name from the applicant's data
        sentences =[]
        applicant_company = job_experience_value['designation']
        applicant_duration = job_experience_value['duration']

        # sentences.append(job_experience_value['description'])

        # Apply the function to the extracted company name
        industry, category = get_industry_for_company(applicant_company)

        # Apply the function to get the industry match result
        industry_match_result = get_industry_for_company(applicant_company)

        # Assign score based on matching result and category
        score = assign_score(industry_match_result, category)

        # Calculate duration in months
        duration_in_months = calculate_duration_in_months(applicant_duration)

        total_duration += duration_in_months

    #*********************************************************************************
        sentences.append(str(job_experience_value['description']))
        print(f"sen: {sentences}")
        check_text = get_unique_words_from_sentences(sentences)
        similarity_score = calculate_similarity(job_description, check_text)

        sim_s += similarity_score * duration_in_months * 100

        print("*_*_*_*_:" )
        print(sim_s)

        score = score * duration_in_months

        total_score += score

        # Display the match result, industry, category, and score
        print("\nMatch Result:")
        print(f"Industry Match for '{applicant_company}' in '{comp_industry}': {industry_match_result}")
        print(f"Industry: {industry}")
        print(f"Category: {category}")
        print(f"Score: {score}")
        print(f"Duration: {duration_in_months}")
        print("FIn sim score :" )
        # print(duration_in_months)
        print(similarity_score)


    # Display unknown matches
    # print("\nUnknown Matches:")
    # print(unknown_matches)
    # print(f"Ratio: {(total_score / 36000) * 100}")
    print(f"Duration Total: {total_duration}")
    # print(f"Sim_s Total: {sim_s/total_duration}")
    print(f"Total: {total_score}")
    min_experience_score, max_experience_score = calculate_experience_range_score(*job_experience_range,comp_vertical)
    print(min_experience_score)
    print(max_experience_score)
    min_required = job_experience_range[0] * 12
    max_required = job_experience_range[1] * 12
    print(f"Duration Max: {min_required}")
    print(f"Duration Min: {max_required}")

    tot_experience = applicant_data['total_experience'] * 12

    if min_required <= tot_experience <= max_required:
        final_Score = 1
    elif  min_required <= total_duration <= max_required:
        final_Score = 1
    elif min_required > total_duration:
        if(min_required - total_duration >= 12):
            final_Score = (0.75 * total_score) / min_experience_score
        elif(min_required - total_duration >= 0):
            final_Score = (0.5 * total_score) / min_experience_score
        else:
            final_Score = 0
    elif max_required < total_duration:
        print(total_duration - max_required)
        loop_val1 = total_duration - max_required
        if(12 <= loop_val1 <= 24):
            final_Score = (0.5 * total_score) / max_experience_score
        elif(24 <= loop_val1 <= 36):
            final_Score = (0.25 * total_score) / max_experience_score
        elif(loop_val1 <= 12):
            final_Score = (0.75 * total_score) / max_experience_score
        elif(36 <= loop_val1 <= 48):
            final_Score = (0.05 * total_score) / max_experience_score
        elif(loop_val1 > 48):
            final_Score = 0

    print(f"FINAL SCORE: {final_Score}")




    # print(f"Slist size: {len(sentences)}")
    # result2 = get_unique_words_from_sentences(sentences)
    fsum = calculate_similarity(extracted_raw_data(applicant_data['raw_data']),job_description) * 100
    print(f"fsum:{fsum}")
    print(f"tsum:{total_duration}")

    # similarity_score = calculate_similarity(job_description, result2)
    # similarity_score = similarity_score * 100
    # print(f"Similarity Score: {similarity_score}%")
    # if sim_s != 0:
    #   similarity_score = (sim_s / total_duration)
    # else:
    #   similarity_score = 0  
    
    # similarity_score += calculate_similarity_score(similarity_score)
    #*****************************************************************************
     # Mock data for job description
    # job_skills = set(['Python', 'Django', 'Linux', 'Programming', 'C'])
    job_experience_required = 2  # Minimum required experience in years


    # Calculate skill match percentage

    jd_skills = extract_skills(job_description,skills_file=None,job_skills=job_skills)
    print(f"JDSKILLS:{jd_skills}")
    applicant_skills = set(skill.lower() for skill in applicant_data['skills'])
    job_skills = set(skill.lower() for skill in jd_skills)

    skill_match_percentage = (len(applicant_skills.intersection(job_skills)) / len(job_skills)) * 100


    # Check if experience matches
    experience_match = final_Score * 100

    # Check if location matches
    location_match = location_matcher(applicant_data['current_city'],job_location)

    # Calculate overall score
    overall_score = calculate_overall_score(skill_match_percentage, experience_match, location_match, fsum)

    # Visualize the analysis
    result = {
        'Skills Match Percentage': skill_match_percentage,
        'Experience Match': experience_match,
        'Location Match': location_match,
        'Overall Score': overall_score
    }

    #*****************************************************************************
    print(result)
    # result = analyze_job_match(applicant_data, job_description,total_score)

    analysis_results = [
        result['Skills Match Percentage'],
        result['Experience Match'],
        result['Location Match'],
        result['Overall Score'],
        fsum
    ]
    print(analysis_results)
    return analysis_results



    
