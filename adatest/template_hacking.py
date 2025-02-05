def templatizev3(string, dictionary=None):
    good =  '|'.join(['good', 'really good', 'great', 'excellent', 'awesome', 'amazing', 'very good'])
    name = 'John|Lucy|Bob|Mary|Ivan|Luke|Matthew'
    if dictionary is None:
        good_fillin = good
        name_fill_in = 'John'
    else:
        good_fillin = 'positive_adj'
        name_fill_in = '{first_name}'
    examples = [
        ('This is a good movie', '{%s} is a good movie' % '|'.join(['This', 'this', 'It', 'it'])),
        ('John was last seen in Idaho', '%s {%s} in Idaho' % (name_fill_in, '|'.join(['was last seen' , 'was last observed', 'was seen']))),
        ('This is a good movie', '{%s} is a {%s} {%s}' % ('|'.join(['This', 'this', 'It', 'it']),
                                                          good_fillin,
                                                    '|'.join(['movie', 'film', 'book', 'product']))),
        ('I like playing golf.', 'I like {%s}.' % '|'.join(['playing golf', 'dancing', 'swimming', 'exercising']))
    ]
    np.random.shuffle(examples)
    prompt = ''
    if dictionary is not None:
        prompt = 'dictionary:\n'
        prompt += 'first_name: %s\n' % name
        prompt += 'positive_adj: %s\n' % good
        for k, v in dictionary.items():
            prompt += '%s: %s\n' % (k, '|'.join(v))
        prompt += '-----\n'
    for x, template in examples:
        prompt += 'input: %s\noutput: %s\n' % (x, template)
    prompt += 'input: %s\noutput:' % string
    return prompt

# Use:
# original_text = 'John and I went to see a movie the other night, it was good'
# original_text = 'I\'m surprised at how well it works'
# prompt = templatizev3(original_text)
# response = openai.Completion.create(
#     engine="davinci-msft", prompt=prompt, max_tokens=100,
#     temperature=0.95, n=10, stop="\n"
#         )
# lines = [choice["text"] for choice in response["choices"]]
# print('Original: %s' % original_text)
# for line in lines:
#     print(line)
#     print()

# print()
# print('With dictionary:')
# prompt = templatizev3(original_text, {'event': ['movie', 'play', 'performance']})
# response = openai.Completion.create(
#     engine="davinci-msft", prompt=prompt, max_tokens=100,
#     temperature=0.95, n=10, stop="\n"
#         )
# lines = [choice["text"] for choice in response["choices"]]
# print('Original: %s' % original_text)
# for line in lines:
#     print(line)
#     print()

def suggest_fill_in(template, key, existing_fill_ins={}):
        examples = [
        ('This is a good movie', {'This': ['this', 'It', 'it']}),
        ('John was last seen in Idaho', {'was last seen': ['was last observed', 'was seen']}),
        ('This is a good movie', {'This': ['this', 'It', 'it'],
                                                        'good': ['really good', 'great', 'excellent'],
                                                        'movie': ['film', 'book', 'product']}),
        ('I like playing golf.',  {'playing golf': ['dancing', 'swimming', 'exercising']})
        ]
        np.random.shuffle(examples)
        prompt = ''
        for x, reps in examples:
            prompt += 'input: %s\nReplacements:\n' % (x)
            for k, v in reps.items():
                prompt += '%s: %s\n' % (k, '|'.join(v))
            prompt += '---\n'

        prompt += 'input: %s\nReplacements:\n' % template
        for k, v in existing_fill_ins.items():
            prompt += '%s: %s\n' % (k, '|'.join(v))
        prompt += '%s:' % key
        return prompt


# Use:
# original_text = 'Scott Lundberg is an engineer at Microsoft'
# rep = 'is an engineer'
# fillins = {
# #     'like': ['love', 'enjoy', 'really like']
# }
# prompt = suggest_fill_in(original_text, rep, fillins)
# # print(prompt)
# response = openai.Completion.create(
#     engine="davinci-msft", prompt=prompt, max_tokens=100,
#     temperature=0.95, n=5, stop="\n"
#         )
# lines = [choice["text"] for choice in response["choices"]]
# print('Original: %s' % original_text)
# print('Suggest replacements for "%s"' % rep)
# for line in lines:
#     print(line)
