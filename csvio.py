import csv
import model
import numpy as np

# encounter_id,patient_nbr,race,gender,age,
# weight, admission_type_id,discharge_disposition_id,admission_source_id,time_in_hospital,
# payer_code, medical_specialty,num_lab_procedures,num_procedures,num_medications,
# number_outpatient,number_emergency,number_inpatient,diag_1,diag_2,
# diag_3,number_diagnoses,max_glu_serum,A1Cresult,metformin,
# repaglinide,nateglinide,chlorpropamide,glimepiride,acetohexamide,
# glipizide,glyburide,tolbutamide,pioglitazone,rosiglitazone,
# acarbose,miglitol,troglitazone,tolazamide,examide,
# citoglipton,insulin,glyburide-metformin,glipizide-metformin,glimepiride-pioglitazone,
# metformin-rosiglitazone,metformin-pioglitazone,change,diabetesMed,readmitted


indexes = {}
last = {}


def lookup(i: int, val):
    if val not in indexes[i]:
        indexes[i][val] = last[i]
        last[i] += 1
    return indexes[i][val]

# x: not used
# s: string
# i: already indexed
# d: digit
# f: float


form = ['x', 'x', 's', 's', 's', 'x', 'i', 'i', 'i', 'd',
        's', 's', 'd', 'd', 'd', 'd', 'd', 'd', 's', 'x',
        'x', 'd', 's', 's', 's', 's', 's', 's', 's', 's',
        's', 's', 's', 's', 's', 's', 's', 's', 's', 's',
        's', 's', 's', 's', 's', 's', 's', 's', 's', 's']

patients = []

test_form = ['s', 's', 's', 's', 's']


def test_get_data(filename):
    with open(filename, 'r') as f:
        ps = []
        for i in range(len(test_form)):
            indexes[i] = {}
            last[i] = 1
        for line in f:
            p = []
            for i, (tag, val) in enumerate(zip(test_form, line[:-1].split(','))):
                if tag == 's':
                    p.append(lookup(i, val))
            ps.append(p)
        return np.asarray(ps, dtype=np.uint8)


class Storage:
    def __init__(self):
        self.mappings = []
        self.last_index = []
        self.titles = None
        self.column_num = None
        self.row_num = None
        self.data = []

    def lookup(self, i: int, val):
        if val not in self.mappings[i]:
            self.mappings[i][val] = self.last_index[i]
            self.last_index[i] += 1
        return self.mappings[i][val]

    def load_csv(self, csvreader, titled=True):
        if titled:
            self.titles = next(csvreader)

        for line in csvreader:
            if self.column_num is None:
                self.column_num = len(line)
                self.mappings = [{} for j in range(self.column_num)]
                self.last_index = [0] * self.column_num

            p = []
            for i, val in enumerate(line):
                # lookup
                if val not in self.mappings[i]:
                    self.mappings[i][val] = self.last_index[i]
                    self.last_index[i] += 1
                p.append(self.mappings[i][val])

            self.data.append(p)

        self.data = np.asarray(self.data, dtype=np.uint8)
        self.row_num = self.data.shape[0]

    def re_mapping(self, index: int, mapping: dict):
        new_mapping = {}
        for mapped_index, val in self.mappings[index].items():
            new_mapping[mapping[mapped_index] if mapped_index in mapping else mapped_index] = val
        self.mappings[index] = new_mapping



def preprocessing(filename):
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        datas = []
        titles = next(lines)
        for line in lines:
            p = []
            for i, val in enumerate(line):
                # useless field
                # 0, 1,   unique or nearly unique, hard to classify
                # 5,      null value too much
                # 39, 40  all of one value
                # 19, 20  2nd and 3rd diagnosis
                if i in [0, 1, 5, 19, 20, 39, 40]:
                    continue

                # numeric field to interval
                # 12, 14  number of lab procedures/medications
                elif i in [12, 14]:
                    seg = int(val) // 10
                    p.append('[%d-%d)' % (seg * 10, (seg + 1) * 10))

                else:
                    p.append(val)


def get_data(filename):
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        patients = []

        titles = next(lines)
        for title in titles:
            indexes[title] = {}
            last[title] = 1

        for line in lines:
            p = []
            for i, (tag, val) in enumerate(zip(form, line)):
                if tag == 'x':
                    continue
                elif tag == 'i':
                    p.append(int(val))
                elif tag == 'd':
                    p.append(int(val))
                elif tag == 'f':
                    p.append(float(val))
                elif tag == 's':
                    p.append(lookup(titles[i], val))
            # special field
            discharge_type_id = p[4]
            if discharge_type_id in [11, 13, 14, 19, 20, 21]:
                continue

            diag_1 = line[18]
            if '250' not in diag_1:
                continue

            patients.append(p)

        return np.asarray(patients, dtype=np.uint8)


def generate(data: np.ndarray):
    return data[:, :-1], data[:, -1]

