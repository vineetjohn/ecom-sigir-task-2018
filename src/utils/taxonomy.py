class Taxonomy:
    def __init__(self):
        self.root = TaxonomyNode(-1)
        self.nodes = set()


class TaxonomyNode:
    def __init__(self, id):
        self.id = id
        self.parent = None
        self.children = dict()
