from openpyxl import load_workbook
from openpyxl import Workbook
import pickle

# Data format:
# PMID Title Abstract BROAD-CATEGORY
broad_category_data = []
# PMID Title Abstract HEALTH-CODES
health_codes_data = []
# PMID Title Abstract Research-Activity-Code
research_activity_code_data = []

total_row_count = 0
for i in range(1, 24):              #number of files = 23
    filename = "data/" + str(i) + ".xlsx"       #file names are modified to go from 0-23
    wb = load_workbook(filename = filename, read_only=True)
    ws = wb['PubMed2XL']            #worksheet "PubMed2XL" in the loaded workbook

    label_row = tuple(ws.rows)[0]
    num_columns = len(label_row)

    pmid_column_id = -1
    title_column_id = -1
    abstract_column_id = -1
    broad_category_column_id = -1
    health_code_column_id = -1
    research_activity_code_column_id = 0
    for j in range(num_columns):
        if label_row[j].value is None: continue
        if label_row[j].value.lower().strip() == "pmid":
            pmid_column_id = j
        if label_row[j].value.lower().strip() == "article title":
            title_column_id = j
        if label_row[j].value.lower().strip() == "abstract":
            abstract_column_id = j
        if label_row[j].value.lower().strip() == "broad disease category":
            broad_category_column_id = j
        if label_row[j].value.lower().strip() == "health codes" or label_row[j].value.lower().strip() == "health code":
            health_code_column_id = j
        if label_row[j].value.lower().strip() == "research activity code":
            research_activity_code_column_id = j


    if (pmid_column_id == -1) or (title_column_id == -1) or (abstract_column_id == -1) or (broad_category_column_id == -1) or (health_code_column_id == -1) or (research_activity_code_column_id == -1):
        print "CORRUPTED FILE " + filename
        for j in range(num_columns):
            print label_row[j].value

    for row in tuple(ws.rows):
        pmid_str = row[pmid_column_id].value
        if pmid_str is None: continue

        total_row_count += 1
        pmid = 0
        try:
            pmid = int(pmid_str)
        except ValueError:
            continue

        title = row[title_column_id].value
        abstract = row[abstract_column_id].value

        broad_category = ""
        health_code = ""
        research_activity_code = ""
        if (broad_category_column_id != -1):
            broad_category = row[broad_category_column_id].value
        if (health_code_column_id != -1):
            health_code = row[health_code_column_id].value
        if (research_activity_code_column_id != -1):
            research_activity_code = row[research_activity_code_column_id].value

        if (broad_category is not None) and (broad_category != "") and (abstract is not None) and (abstract != ""):
            # print title + " ::: " + broad_category
            broad_category_data.append([pmid_str, title, abstract, broad_category])

        if (health_code is not None) and (health_code != "") and (abstract is not None) and (abstract != ""):
            # print title + " ::: " + health_code
            health_codes_data.append([pmid_str, title, abstract, health_code])

        if (research_activity_code is not None) and (research_activity_code != "") and (abstract is not None) and (abstract != ""):
            # print title + " ::: " + research_activity_code
            research_activity_code_data.append([pmid_str, title, abstract, research_activity_code])


print "Finished parsing data. Total data-set: " + str(total_row_count)
print "Broad category training set size: " + str(len(broad_category_data))
print "Health codes training set size: " + str(len(health_codes_data))
print "Research activity codes training set size: " + str(len(research_activity_code_data))

with open('output/broad_category_data1.obj', 'wb') as fp:
    pickle.dump(broad_category_data, fp)

with open('output/health_codes_data1.obj', 'wb') as fp:
    pickle.dump(health_codes_data, fp)

with open('output/research_activity_code_data1.obj', 'wb') as fp:
    pickle.dump(research_activity_code_data, fp)


# reading the data back
# with open ('output/broad_category_data.obj', 'rb') as fp:
#     broad_category_data = pickle.load(fp)
