class FormatPrompt:
    @staticmethod
    def format(state, text, choices):
        options = "\n".join( f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))

        try:
          prompt = f"""STATE: {state}
          INSTRUCTION: {text}
          OPTIONS: {options}
          ANSWER:"""
          return prompt
        except:
          raise ValueError("Either question or instructions must be provided")