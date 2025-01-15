class AbstentionScorer:
    def __init__(self):
        self.invalid_mentions = [
            "I could not find any information",
            "The search results do not provide",
            "There is no information",
            "There are no search results",
            "there are no provided search results",
            "not provided in the search results",
            "is not mentioned in the provided search results",
            "There seems to be a mistake in the question",
            "Not sources found",
            "No sources found",
            "Try a more general question",
            "Unfortunately,",
            "There doesn't seem to be",
            "There does not seem to be",
            "I do not",
            "I don't",
            "**No relevant",
            "I'm afraid",
            "I am afraid",
            "I apologize,",
            "I'm sorry",
            "I am sorry"
            "Sorry",
            "provide more",
            "I am not familiar with",
            "I'm not familiar with",
        ]

    def is_response_abstained(self, generation):
        """
        Detect if the generation is an abstention.
        """


        for x in self.invalid_mentions:
            # print(generation)
            try:
                if generation and x in generation:
                    return True
            except Exception as e:
                return True

        return False
