
import io
from io import StringIO
import os
import re
import json
import dateparser
import nltk
import pandas as pd
import docx2txt
from datetime import datetime
from dateutil import relativedelta
from . import constants as cs
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.pdfparser import PDFParser
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def extract_text_from_pdf(pdf_path):
    '''
    Helper function to extract the plain text from .pdf files

    :param pdf_path: path to PDF file to be extracted (remote or local)
    :return: iterator of string of extracted text
    '''
    # https://www.blog.pythonlibrary.org/2018/05/03/exporting-data-from-pdfs-with-python/
    if not isinstance(pdf_path, io.BytesIO):
        # extract text from local pdf file
        with open(pdf_path, 'rb') as fh:
            try:
                for page in PDFPage.get_pages(
                        fh,
                        caching=True,
                        check_extractable=True
                ):
                    resource_manager = PDFResourceManager()
                    fake_file_handle = io.StringIO()
                    converter = TextConverter(
                        resource_manager,
                        fake_file_handle,
                        laparams=LAParams()
                    )
                    page_interpreter = PDFPageInterpreter(
                        resource_manager,
                        converter
                    )
                    page_interpreter.process_page(page)

                    text = fake_file_handle.getvalue()
                    yield text

                    # close open handles
                    converter.close()
                    fake_file_handle.close()
            except PDFSyntaxError:
                return
    else:
        # extract text from remote pdf file
        try:
            for page in PDFPage.get_pages(
                    pdf_path,
                    caching=True,
                    check_extractable=True
            ):
                resource_manager = PDFResourceManager()
                fake_file_handle = io.StringIO()
                converter = TextConverter(
                    resource_manager,
                    fake_file_handle,
                    laparams=LAParams()
                )
                page_interpreter = PDFPageInterpreter(
                    resource_manager,
                    converter
                )
                page_interpreter.process_page(page)

                text = fake_file_handle.getvalue()
                yield text

                # close open handles
                converter.close()
                fake_file_handle.close()
        except PDFSyntaxError:
            return


def get_number_of_pages(file_name):
    try:
        if isinstance(file_name, io.BytesIO):
            # for remote pdf file
            count = 0
            for page in PDFPage.get_pages(
                        file_name,
                        caching=True,
                        check_extractable=True
            ):
                count += 1
            return count
        else:
            # for local pdf file
            if file_name.endswith('.pdf'):
                count = 0
                with open(file_name, 'rb') as fh:
                    for page in PDFPage.get_pages(
                            fh,
                            caching=True,
                            check_extractable=True
                    ):
                        count += 1
                return count
            else:
                return None
    except PDFSyntaxError:
        return None


def extract_text_from_docx(doc_path):
    '''
    Helper function to extract plain text from .docx files

    :param doc_path: path to .docx file to be extracted
    :return: string of extracted text
    '''
    try:
        temp = docx2txt.process(doc_path)
        text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
        return ' '.join(text)
    except KeyError:
        return ' '


def extract_text_from_doc(doc_path):
    '''
    Helper function to extract plain text from .doc files

    :param doc_path: path to .doc file to be extracted
    :return: string of extracted text
    '''
    try:
        try:
            import textract
        except ImportError:
            return ' '
        text = textract.process(doc_path).decode('utf-8')
        return text
    except KeyError:
        return ' '


def extract_text(file_path, extension):
    '''
    Wrapper function to detect the file extension and call text
    extraction function accordingly

    :param file_path: path of file of which text is to be extracted
    :param extension: extension of file `file_name`
    '''
    text = ''
    if extension == '.pdf':
        for page in extract_text_from_pdf(file_path):
            text += ' ' + page
    elif extension == '.docx':
        text = extract_text_from_docx(file_path)
    elif extension == '.doc':
        text = extract_text_from_doc(file_path)
    return text


def extract_entity_sections_grad(text):
    '''
    Helper function to extract all the raw text from sections of
    resume specifically for graduates and undergraduates

    :param text: Raw text of resume
    :return: dictionary of entities
    '''
    text_split = [i.strip() for i in text.split('\n')]
    # sections_in_resume = [i for i in text_split if i.lower() in sections]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(cs.RESUME_SECTIONS_GRAD)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.RESUME_SECTIONS_GRAD:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)

    # entity_key = False
    # for entity in entities.keys():
    #     sub_entities = {}
    #     for entry in entities[entity]:
    #         if u'\u2022' not in entry:
    #             sub_entities[entry] = []
    #             entity_key = entry
    #         elif entity_key:
    #             sub_entities[entity_key].append(entry)
    #     entities[entity] = sub_entities

    # pprint.pprint(entities)

    # make entities that are not found None
    # for entity in cs.RESUME_SECTIONS:
    #     if entity not in entities.keys():
    #         entities[entity] = None
    return entities


def extract_entities_wih_custom_model(custom_nlp_text):
    '''
    Helper function to extract different entities with custom
    trained model using SpaCy's NER

    :param custom_nlp_text: object of `spacy.tokens.doc.Doc`
    :return: dictionary of entities
    '''
    entities = {}
    for ent in custom_nlp_text.ents:
        if ent.label_ not in entities.keys():
            entities[ent.label_] = [ent.text]
        else:
            entities[ent.label_].append(ent.text)
    for key in entities.keys():
        entities[key] = list(set(entities[key]))
    return entities


def get_total_experience(experience_list):
    '''
    Wrapper function to extract total months of experience from a resume

    :param experience_list: list of experience text extracted
    :return: total months of experience
    '''
    exp_ = []
    for line in experience_list:
        experience = re.search(
            r'(?P<fmonth>\w+.\d+)\s*(\D|to)\s*(?P<smonth>\w+.\d+|present)',
            line,
            re.I
        )
        if experience:
            exp_.append(experience.groups())
    total_exp = sum(
        [get_number_of_months_from_dates(i[0], i[2]) for i in exp_]
    )
    total_experience_in_months = total_exp
    return total_experience_in_months


def get_number_of_months_from_dates(date1, date2):
    '''
    Helper function to extract total months of experience from a resume

    :param date1: Starting date
    :param date2: Ending date
    :return: months of experience from date1 to date2
    '''
    if date2.lower() == 'present':
        date2 = datetime.now().strftime('%b %Y')
    try:
        if len(date1.split()[0]) > 3:
            date1 = date1.split()
            date1 = date1[0][:3] + ' ' + date1[1]
        if len(date2.split()[0]) > 3:
            date2 = date2.split()
            date2 = date2[0][:3] + ' ' + date2[1]
    except IndexError:
        return 0
    try:
        date1 = datetime.strptime(str(date1), '%b %Y')
        date2 = datetime.strptime(str(date2), '%b %Y')
        months_of_experience = relativedelta.relativedelta(date2, date1)
        months_of_experience = (months_of_experience.years
                                * 12 + months_of_experience.months)
    except ValueError:
        return 0
    return months_of_experience


def extract_entity_sections_professional(text):
    '''
    Helper function to extract all the raw text from sections of
    resume specifically for professionals

    :param text: Raw text of resume
    :return: dictionary of entities
    '''
    text_split = [i.strip() for i in text.split('\n')]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) \
                    & set(cs.RESUME_SECTIONS_PROFESSIONAL)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.RESUME_SECTIONS_PROFESSIONAL:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)
    return entities


def extract_email(text):
    '''
    Helper function to extract email id from text

    :param text: plain text extracted from resume file
    '''
    email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None


def extract_name(nlp_text, matcher):
    '''
    Helper function to extract name from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param matcher: object of `spacy.matcher.Matcher`
    :return: string of full name
    '''
    pattern = [cs.NAME_PATTERN]

    matcher.add('NAME', None, *pattern)

    matches = matcher(nlp_text)

    for _, start, end in matches:
        span = nlp_text[start:end]
        if 'name' not in span.text.lower():
            return span.text


def extract_mobile_number(text, custom_regex=None):
    '''
    Helper function to extract mobile number from text

    :param text: plain text extracted from resume file
    :return: string of extracted mobile numbers
    '''
    # Found this complicated regex on :
    # https://zapier.com/blog/extract-links-email-phone-regex/
    # mob_num_regex = r'''(?:(?:\+?([1-9]|[0-9][0-9]|
    #     [0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|
    #     [2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|
    #     [0-9]1[02-9]|[2-9][02-8]1|
    #     [2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|
    #     [2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{7})
    #     (?:\s*(?:#|x\.?|ext\.?|
    #     extension)\s*(\d+))?'''
    if not custom_regex:
        mob_num_regex = r'''(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)
                        [-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'''
        phone = re.findall(re.compile(mob_num_regex), text)
    else:
        phone = re.findall(re.compile(custom_regex), text)
    if phone:
        number = ''.join(phone[0])
        return number


def extract_skills(nlp_text, noun_chunks, skills_file=None):
    '''
    Helper function to extract skills from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :param noun_chunks: noun chunks extracted from nlp text
    :return: list of skills extracted
    '''
    tokens = [token.text for token in nlp_text if not token.is_stop]
    if not skills_file:
        data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), 'skills.csv')
        )
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
    print(skillset)
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def cleanup(token, lower=True):
    if lower:
        token = token.lower()
    return token.strip()


def extract_education(nlp_text):
    '''
    Helper function to extract education from spacy nlp text

    :param nlp_text: object of `spacy.tokens.doc.Doc`
    :return: tuple of education degree and year if year if found
             else only returns education degree
    '''
    edu = {}
    # Extract education degree
    try:
        for index, text in enumerate(nlp_text):
            for tex in text.split():
                tex = re.sub(r'[?|$|.|!|,]', r'', tex)
                if tex.upper() in cs.EDUCATION and tex not in cs.STOPWORDS:
                    edu[tex] = text + nlp_text[index + 1]
    except IndexError:
        pass

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(cs.YEAR), edu[key])
        if year:
            education.append((key, ''.join(year.group(0))))
        else:
            education.append(key)
    return education




def extract_experience(raw_text):
    '''
    Helper function to extract experience from resume text

    :param resume_text: Plain resume text
    :return: list of experience
    '''
    
    json_file_path = 'pyresparser\job-titles.json'

    newdict = {}

    # Create a set to store keys with duplicate values
    duplicate_keys = set()

    # Create a set to store unique values
    unique_values = set()

    output_string = StringIO()
    with open(raw_text, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    pdf_content = output_string.getvalue().lower()

    print(pdf_content)
    # pdf_content = extracted_text
    # print(output_string.getvalue())
    # Find the indices of "Experience" and "Projects" to extract the relevant text
    # start_index = pdf_content.find("work experience")
    start_index = pdf_content.find("experience")
    pdf_content2 = pdf_content[start_index+1:]
    end_index_project = pdf_content2.find("projecta")
    end_index_certificate = pdf_content2.find("certification")
    end_index_exp1 = pdf_content2.find("leadership experience")
    end_index_exp2 = pdf_content2.find("another experience")
    end_index_edu = pdf_content2.find("education")



    # Get the minimum non-negative index among project, certificate, and language
    end_index_options = [index for index in [end_index_project, end_index_certificate, end_index_exp1, end_index_exp2, end_index_edu] if index >= 0]

    if end_index_options:
        # If any of the words are found, set end_index to the minimum non-negative index
        end_index = min(end_index_options)+start_index
    else:
        # If none of the words are found, set end_index to the end of the document
        end_index = len(pdf_content)

    # Extract the text between "Experience" and "Projects"
    experience_text = pdf_content[start_index:end_index].split('\n', 1)[1]

    lines = [line for line in experience_text.split('\n') if line.strip() != '']

    # Print the extracted experience text
    # print(lines)
    # print(experience_text)

    existing_text=''''''

    for element in lines:
        existing_text += f"\n{element}"

    # print(existing_text)
    # print(type(lines))

    def add_description(result_dict):
      keys = list(result_dict.keys())

      # Check if the keys list is not empty
      if keys:
          last_key = keys[-1]

          for i in range(len(keys)-1):
              current_key = keys[i]
              next_key = keys[i+1]

              #latest edit
              line_before = extract_company_name(result_dict[current_key]['line_before'])
              result_dict[current_key]['line_before']= line_before
              #end

              # Find the indices of current and next matched texts
              current_index = existing_text.find(result_dict[current_key]['matched_text'])
              next_index = existing_text.find(result_dict[next_key]['matched_text'])
              print(f"cur: {current_index}")
              print(f"next: {next_index}")

              # Extract the description between current and next matched texts
              description = existing_text[current_index + len(result_dict[current_key]['matched_text']):next_index].strip()

              if current_index > next_index :
                  description = existing_text[current_index + len(result_dict[current_key]['matched_text']):].strip()

              # Update the description in the result_dict
              result_dict[current_key]['description'] = description

          # Handle the description for the last key separately
          last_index = existing_text.find(result_dict[last_key]['matched_text'])
          description_last = existing_text[last_index + len(result_dict[last_key]['matched_text']):].strip()
          result_dict[last_key]['description'] = description_last
          line_before = extract_company_name(result_dict[last_key]['line_before'])
          result_dict[last_key]['line_before']= line_before

    def extract_company_name(input_string):
        # Define a regular expression pattern to match different formats of organization names
        patterns = [
            re.compile(r'\borganization\b\s*[:-]?\s*[^\d.,]+\.?\s*([^\d.,]+)\.?\s*'),
            re.compile(r'[^.,\w]([\w\s]+),?\s*([^\d.,]+)\b'),
            re.compile(r'([^\d.,]+)\s*technology', re.IGNORECASE),
            re.compile(r'orgnication\s*[:-]?\s*([^\d.,]+)'),
            re.compile(r'([^\d.,]+)\scements', re.IGNORECASE)
        ]

        # Search for each pattern in the input string
        for pattern in patterns:
            match = pattern.search(input_string)
            if match:
                return match.group(1).strip()

        # If no match is found, return None or handle it as needed
        return input_string

    def find_matches(json_file_path, text):
        result_dict = {}
        result_dict2 = {}

        # Load JSON file
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Iterate through the text
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # print(f"Processing line {i + 1}: {line}")
            # Check if the line contains any job title from the JSON file
            for job_title in data["job-titles"]:
                # print(f"Chefcking for {job_title}")
                # print(job_title)
                if job_title in line.lower():
                    # print(f"found: {job_title}")
                    # Get the line before and after the matched text
                    previous_line = lines[i - 1] if i - 1 >= 0 else None
                    next_line = lines[i + 1] if i + 1 < len(lines) else None
                    next_line2 = lines[i + 2] if i + 2 < len(lines) else None

                    # Add to the dictionary if any of the lines contain a date
                    if has_date(next_line) or has_date(line) or has_date(previous_line):
                        line_after = (next_line + next_line2) if (next_line is not None and next_line2 is not None) else next_line or next_line2 or ''
                        result_dict[job_title] = {
                            'designation': job_title,
                            'matched_text': line,
                            'line_before': previous_line,
                            'line_after': line_after
                        }
                    else:
                        result_dict2[job_title] = {
                            'matched_text': line,
                            'line_before': previous_line,
                            'line_after': next_line
                        }


        return result_dict

    def has_date(text):
        # Check if the text contains a date in various formats
        if text is None:
            return False

        formats = [
            r'(?:\b\d{1,2}-[a-zA-Z]{3,}-\d{4}\b\s*to\s*\b\d{1,2}-[a-zA-Z]{3,}-\d{4}\b)'  # additional format: 17-jan-2023 to 15-sep-2023
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*-\s*\d{4}\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}\b',
            r'\b\d{1,2}/\d{4}\s*-\s*\d{1,2}/\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\s*-\s*\d{1,2}/\d{1,2}/\d{4}\b',
            r'(?:\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\'’](?:\d{2}|\d{4})\s*-\s*\b(?:till\s*date|(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\'’](?:\d{2}|\d{4})\b))',
            
            r'\b\d{1,2}-[a-zA-Z]{3,}-\d{4}\b',  # e.g., 15-sep-2023
        ]

        for date_format in formats:
            if re.search(date_format, text, re.IGNORECASE):
                return True

        return False

    def clean_date(text):
        formats = [
            r'(?:\b\d{1,2}-[a-zA-Z]{3,}-\d{4}\b\s*to\s*\b\d{1,2}-[a-zA-Z]{3,}-\d{4}\b)',  # additional format: 17-jan-2023 to 15-sep-2023
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4}\s*–\s*(?:january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4}\b',  # April 2021 – September 2021
            
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}\s*to\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}\b',  # Oct 2020 to Jan 2023
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*-\s*\d{4}\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}\b',
            r'\b\d{1,2}/\d{4}\s*-\s*\d{1,2}/\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\s*-\s*\d{1,2}/\d{1,2}/\d{4}\b',
            r'(?:\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\'’](?:\d{2}|\d{4})\s*-\s*\b(?:till\s*date|(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\'’](?:\d{2}|\d{4})\b))',
            r'\b\d{1,2}-[a-zA-Z]{3,}-\d{4}\b',  # e.g., 15-sep-2023
        ]

        current_check = [
            r'to present',
            r'- present',
        ]

        for date_format in formats:
            match = re.search(date_format, text, re.IGNORECASE)
            if match:
                for current_pattern in current_check:
                  if re.search(current_pattern, text, re.IGNORECASE):
                      clean_date = match.group(0) + " to present"
                      return clean_date
                return match.group(0)

        return text


    matches = find_matches(json_file_path, existing_text)
    print("*********************")
    # print(matches)
    for key, value in matches.items():
        # Check if 'line_before' and 'line_after' keys exist
        if 'line_before' in value and 'line_after' in value and 'matched_text' in value:
            # Extract dates from 'line_before' and 'line_after'
            date_before = value['line_before'].strip()
            date_after = value['line_after'].strip()
            date_in_matched = value['matched_text'].strip()

            # Check if either line contains a date
            if has_date(date_in_matched):
                newdict[key] = {
                    'date': date_in_matched,
                }
            elif has_date(date_after):
                newdict[key] = {
                    'date': date_after,
                }
            elif has_date(date_before):
                newdict[key] = {
                    'date': date_before,
                }




    # Iterate through the dictionary
    for key, value in newdict.items():
        # Convert the nested dictionary values to strings for comparison
        value_str = str(value)

        # Check if the value is already in the unique_values set
        # If yes, add the key to the duplicate_keys set
        if value_str in unique_values:
            duplicate_keys.add(key)
        else:
            unique_values.add(value_str)

    for key in duplicate_keys:
        del matches[key]

    add_description(matches)

    for key, value in matches.items():
        matches[key]["duration"] = newdict[key]['date']
        matches[key]["duration"] = clean_date(matches[key]["duration"])

    return matches





def extract_gender(text):
    # Define regular expressions for gender detection
    male_keywords = re.compile(r'\b(?:he|him|his|male|)\b', re.IGNORECASE)
    female_keywords = re.compile(r'\b(?:she|her|hers|female)\b', re.IGNORECASE)

    # Check for male keywords
    if male_keywords.search(text):
        return 'Male'

    # Check for female keywords
    elif female_keywords.search(text):
        return 'Female'

    # Default to 'Unknown' if no gender-related keywords are found
    else:
        return 'Unknown'


def extract_current_city(text):
    file_path = 'pyresparser\worldcities.csv'
    df = pd.read_csv(file_path, encoding='latin-1')
    cities_list = df[df['iso2'] == 'IN']['city'].tolist()
    current_city_pattern = re.compile(r'Current\s*City[:\s]*(\w+)', re.IGNORECASE)
    match = re.search(current_city_pattern, text)

    if match:
        return match.group(1).strip()

    # If no city is found using the pattern, use email and next 6 lines logic
    email = extract_email(text)

    if email:
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if email in line:
                # Check if any cities from the list match the extracted text
                for city in cities_list:
                    if city.lower() in text.lower():
                        return city.strip()
                # If no match in the city list, return "Unknown"
                return "Unknown"

    # If no email or city is found, return "Unknown"
    return "Unknown"



def extract_resume_objective(text):
    objective_pattern = re.compile(r'Objective[:\s]*(.*?)(?:\n|$)', re.IGNORECASE | re.DOTALL)
    match = re.search(objective_pattern, text)
    return match.group(1).strip() if match else None
    # return text

def text_all(text):
    print(text)
    # Testing functions on a sample resume
    return text


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