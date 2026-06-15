class ToFDataLabel:
    class Label:
        def __init__(self, index: int, name: str):
            self.index = index
            self.name = name


    labels = [
        Label(0, 'Normal'),
        Label(1, 'Upstair'),
        Label(2, 'Downstair'),
    ]

    label_cnts = len(labels)

    combo_labels = [label.name for label in labels]