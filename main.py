import csvio
import model
import inspect as ins
import json

def collect():
    hs = {
        'age': {},
        'gender': {},
        'race': {},
        'admission_type': {},
        'diagnosis_1': {},
        'discharge': {}
    }

    for p in model.patients.values():
        for k, v in hs.items():
            attr = ins.getattr_static(p, k)
            if attr in v:
                v[attr] += 1
            else:
                v[attr] = 1

    json.dump(hs, open('problem1.json', 'w'), indent=4, ensure_ascii=False)
    # print(hs)

def see250():
    cnt = 0
    for p in model.patients.values():
        if '250' in p.diagnosis_1:
            cnt += 1
    print(cnt)

if __name__ == '__main__':
    # csvio.lrace('diabetic_data.csv')
    x = csvio.get_data('diabetic_data.csv')
    d, l = csvio.generate(x)
    # see250()
    print(x.shape)
    print(d.shape, l.shape)

