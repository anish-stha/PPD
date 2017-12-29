from sklearn.svm import SVC
from FeatureExtractions import *
from sklearn import svm
from scipy import stats
import cv2
from sklearn.externals import joblib
import numpy

image_show = False

paths = [
            ['Corn/Common Rust/1.jpg',
                'Corn/Common Rust/2.jpg',
                    'Corn/Common Rust/3.jpg'],
            ['Corn/Eyespot/3.jpg',
            'Corn/Eyespot/4.jpg',
            'Corn/Eyespot/5.jpg',
            'Corn/Eyespot/6.jpg',
            'Corn/Eyespot/7.jpg',
            'Corn/Eyespot/8.jpg'],
            ['Corn/Goss Wilt/1.jpg'],
        ['Corn/SouthernCornLeafBlight/1.jpg',
        'Corn/SouthernCornLeafblight/2.jpg',
        'Corn/SouthernCornLeafblight/3.jpg',
        'Corn/SouthernCornLeafblight/4.jpg',
        'Corn/SouthernCornLeafblight/5.jpg',
        'Corn/SouthernCornLeafblight/6.png',
        'Corn/SouthernCornLeafblight/7.png',
        'Corn/SouthernCornLeafblight/8.jpg',
        'Corn/SouthernCornLeafblight/9.jpg']]

def get_areas_ratios(path):
    original = cv2.imread(path)

    image = cv2.resize(original, (480, 800), interpolation=cv2.INTER_CUBIC)
    areas = []
    ratios = []
    #display kmeans cq result
    if image_show:
        cv2.imshow('res2', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    contours = get_features(original)
    for cnt in contours:
        areas.append(cv2.contourArea(cnt))
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(h) / w
        ratios.append(ratio)
        # if area < 2000 or area > 4000:
        #     continue
        #
        # if len(cnt) < 5:
        #     continue
    m_ratios = stats.trim_mean(ratios, 0.1)
    m_areas = stats.trim_mean(areas, 0.1)
    print(m_ratios)
    print(m_areas)
    return [len(contours), m_areas, m_ratios]
#
# X = []
# Y = []
#
# for i in range(len(paths)):
#     for j in range(len(paths[i])):
#         current_path = paths[i][j]
#         print("For path:" + current_path)
#         areas_ratios = get_areas_ratios(current_path)
#         X.append(areas_ratios)
#     Y.append(i)

X = np.array([[463, 100.90700808625337, 1.170991804526277], [57, 129.28723404255319, 1.2804688667503907], [318, 85.4375, 1.3363109919516418], [82, 77.242424242424249, 1.3284831083962099], [66, 32.222222222222221, 1.291655676521577], [11, 1253.1111111111111, 1.4251509497860289], [443, 45.959154929577466, 1.1495815317237756], [199, 67.841614906832305, 1.2767316305936058], [19, 20.0, 1.0171558478463849], [66, 50.203703703703702, 1.6007478671782163], [241, 107.84974093264249, 1.4449764812809758], [101, 87.191358024691354, 2.1828305904383094], [94, 527.72368421052636, 2.0641975111002262], [2166, 29.975201845444062, 1.4694092471204747], [519, 198.99880095923263, 1.4837877171839438], [35, 106.44827586206897, 2.2501108651677137], [11, 74.777777777777771, 1.1203703703703705], [311, 61.626506024096386, 1.6054614670978589], [64, 635.39423076923072, 1.5276230789359464]])
Y = [0,0,0,1,1,1,1,1,1,2,3,3,3,3,3,3,3,3,3]
print(X, Y)
clf = svm.SVC()
clf.fit(X, Y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
joblib.dump(clf, 'trained_data.pkl')
test_data = get_areas_ratios('Corn/SouthernCornLeafBlight/1.jpg')
print(clf.predict([test_data]))