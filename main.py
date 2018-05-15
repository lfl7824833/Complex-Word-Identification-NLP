from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
from sklearn.metrics import mean_squared_error

def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} Test\n".format(language.upper(), len(data.trainset), len(data.devset)))

    #for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline(language, type='classify')

    baseline.train(data.trainset)

    predictions = baseline.test(data.devset)

    gold_labels = [sent['gold_label'] for sent in data.devset]

    report_score(gold_labels, predictions, detailed=True)


    ########################### Regression ###################33
    baseline2 = Baseline(language, type='regression')

    baseline2.train(data.trainset)

    predictions = baseline2.test(data.devset)

    gold_labels2 = [float(sent['gold_prob']) for sent in data.devset]

    print("Probabilistic classification task:\nMSE:", mean_squared_error(gold_labels2, predictions),"\n\n")

if __name__ == '__main__':
    execute_demo('english')
    execute_demo('spanish')


