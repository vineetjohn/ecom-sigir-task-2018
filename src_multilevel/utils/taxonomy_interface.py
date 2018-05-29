class Taxonomy:
    def __init__(self):
        self.root = TaxonomyNode(-1, -1)
        self.nodes = dict()
        self.level_nodes = dict()

    def __repr__(self):
        return str(self.root)

    def add_categories(self, categories):
        parent = self.root
        for index, category in enumerate(categories):
            if not category in parent.children:
                category_node = TaxonomyNode(category, index)
                category_node.parent = parent
                parent.children[category] = category_node
                self.nodes[category] = category_node
            else:
                category_node = self.nodes[category]

            if index not in self.level_nodes:
                self.level_nodes[index] = set()
            self.level_nodes[index].add(category)

            parent = category_node


class TaxonomyNode:
    def __init__(self, id, level):
        self.id = id
        self.parent = None
        self.children = dict()
        self.level = level

    def __repr__(self):
        representation = ["{}=(".format(self.id)]
        for child in self.children:
            representation.append("{}, ".format(self.children[child]))
        representation.append(")")
        return "".join(representation)
