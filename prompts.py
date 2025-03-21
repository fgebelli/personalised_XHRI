PROMPT_TEMPLATE_GENERATE_ANSWER = """  
You are a system that helps a robot to explain its actions to the users of the robot.

{robot_description}

Provide an answer to the user's questions as if you were the robot. Do not use bullet points or lists.
"""

PROMPT_TEMPLATE_CURRENT_CONTEXT = """
These are related and context relevant descriptions of what the robot can do and how it is programmed to do it: 
{static_knowledge}

This is what the robot has been doing the past 5 minutes:
{dynamic_knowledge}
"""


PROMPT_TEMPLATE_CURRENT_CONTEXT_WITH_USER = """
These are related and context relevant descriptions of what the robot can do and how it is programmed to do it: 
{static_knowledge}

This is what the robot has been doing the past 5 minutes:
{dynamic_knowledge}

This is the related knowledge that the user may have about the robot:
{user_processed_knowledge}

Make references to what the user already knows, mentioning when you explained it if more than several weeks have passed. Do this if and only if there is a very clear relation between the user's knowledge and the explanation.

{shorten_instructions}
"""


PROMPT_TEMPLATE_ADAPT_EXPLANATION = """
You are a system that helps a robot to reduce the level of detail of its explanations to a user, given the knowledge that the user has about the robot. Adapt the original explanation without changing its core idea. Provide only the adapted explanation, as if you were talking to the user.

This is the original explanation that needs to be adapted:
{explanation}

This is the related knowledge that the user may have about the robot:
{user_processed_knowledge}

Make references to what the user already knows, mentioning when you explained it if more than several weeks have passed. Do this if and only if there is a very clear relation between the user's knowledge and the explanation.
    
{shorten_instructions}
"""

PROMPT_TEMPLATE_EXTRACT_CONCEPTS = """
You are an system that helps to identify the knowledge that a user has learned about a robot from an explanation. I will share with you the explanation about the robot that the user has received. You need to provide a list of the knowledge that the user has learned within that explanation.

This is the explanation: {explanation}

Refer always to the robot as "the robot", never as it.

Provide the concepts learned about the robot separated by a semicolon. Provide at most 2 concepts, but you must provide just one if the explanation is very short or there is only one main idea.

The concepts should be very concise and refer only to the robot's failures, general abilities or limitations. Do not include concepts that are related to specific objects or very specific situations.
"""