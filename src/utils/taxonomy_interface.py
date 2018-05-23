class Taxonomy:
    def __init__(self):
        self.root = TaxonomyNode(-1)
        self.nodes = dict()

    def __repr__(self):
        return str(self.root)

    def add_categories(self, categories):
        parent = self.root
        for category in categories:
            if not category in parent.children:
                category_node = TaxonomyNode(category)
                category_node.parent = parent
                parent.children[category] = category_node
            else:
                category_node = parent.children[category]
            parent = category_node


class TaxonomyNode:
    def __init__(self, id):
        self.id = id
        self.parent = None
        self.children = dict()

    def __repr__(self):
        representation = ["{}=(".format(self.id)]
        for child in self.children:
            representation.append("{}, ".format(self.children[child]))
        representation.append(")")
        return "".join(representation)
