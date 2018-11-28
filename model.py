

indexes = {
    'gender': {
        '?': -1,
        'Unknown/Invalid': -1,
        'Male': 0,
        'Female': 1,
    },
    'race': {
        '?': -1,
        'Asian': 0,
        'AfricanAmerican': 1,
        'Caucasian': 2,
        'Hispanic': 3,
        'Other': 4
    },
    'age': {},
    'admission_type': {},
    'discharge': {}
}

# so
# 1. home and care: 1, 8
# 2. other hospital: 2, 10, 16, 23, 28, 29
# 3. other institution: 3, 4, 5, 6, 7, 22, 24, 27
# 4. this hospital: 9, 15, 17
# 6. expired: 11
# 7. left patient: 12
# 8. hospice: 13, 14, 19, 20, 21
# 9. unknown: 18, 25, 26

# discharge
# 1. 回家
# 2. 去其他短期医院
# 3. 专业护理设施（SNF）
# 4. 残疾健康组织（ICF）
# 5. 关怀组织
# 6. 其他住院病人护理机构
# 7. 美国医学学会
# 8. 静脉注射
# 9. 住在医院
# 10. 初生婴儿转院
# 11. 死亡
# 12. 仍然患病或门诊类的
# 13. 在家临终关怀
# 14. 医学设备临终关怀
# 15. 在本院允许的swing bed？
# 16. 其他机构门诊
# 17. 本院门诊
# 18. NULL
# 19. 只接受医疗补助临终关怀，家中死亡
# 20. 只接受医疗补助临终关怀，医疗机构死亡
# 21. 只接受医疗补助临终关怀，未知地方死亡
# 22. 其他康复机构
# 23. 长期关怀医院
# 24. 医疗补助认证但医疗保险未认证的护理机构
# 25. 未知
# 26. 未知
# 27. 没有注册的关怀机构
# 28. 精神病医院或医院的精神病科
# 29. 重症医院



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


class Patient(object):

    def __init__(self):
        self.encounter_id = ''
        self.age = 0  # age == x equals age == [x * 10, (x + 1) * 10), x in range(0, 10)
        self.gender = 0
        self.race = 0
        self.admission_type = 0
        self.diagnosis_1 = 0
        self.discharge = 0

    def save(self):
        if self.discharge not in [11, 13, 14, 19, 20, 21]:
            Adapter.add(self)

    @classmethod
    def from_line(cls, line):
        # patient:
        # race, gender, age, admission_type, discharge_type,
        # admission_source, time_in_hospital, payer_code, medical_speciality, num_lab_proc
        # num_proc, num_medication,
        patient = Patient()
        patient.race = indexes['race'][line[2]]
        patient.gender = indexes['gender'][line[3]]
        patient.age = int(line[4][1])
        patient.admission_type = int(line[6])
        patient.discharge = int(line[7])
        patient.diagnosis_1 = line[18]
        patient.save()


class Adapter:
    patients = []
    datas = []
    labels = []

    @classmethod
    def add(cls, patient: Patient):
        cls.patients.append(patient)

    @classmethod
    def add_row(cls, data, label):
        cls.datas.append(data)
        cls.labels.append(label)
