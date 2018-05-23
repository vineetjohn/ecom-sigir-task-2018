from sklearn.feature_extraction.text import TfidfVectorizer


def get_data(input_file_path):
    products, labels = list(), list()

    with open(input_file_path) as input_file:
        for line in input_file:
            [product, category_string] = line.strip().split('\t')
            categories = category_string.strip().split('>')
            products.append(product)
            labels.append(categories[-1])

    return products, labels


def get_product_features(products):
    tfidf_vectorizer = TfidfVectorizer(input=products, strip_accents='unicode')
    features = tfidf_vectorizer.fit_transform(products)

    return features
