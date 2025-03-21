from openai import OpenAI
from prompts import PROMPT_TEMPLATE_GENERATE_ANSWER, PROMPT_TEMPLATE_CURRENT_CONTEXT, PROMPT_TEMPLATE_CURRENT_CONTEXT_WITH_USER, PROMPT_TEMPLATE_EXTRACT_CONCEPTS, PROMPT_TEMPLATE_ADAPT_EXPLANATION
from langchain.prompts import PromptTemplate
import datetime
import time
import os
import pickle as pkl
from scipy import spatial
import math

class Explainer():
    def __init__(self, use_RAG = True, use_heuristic = True, one_stage = False, use_question = False, robot = "nicol", retrieve_thresh = 0.35):
        
        # Robot context per topic
        self.generalized_answers = {
            "asr_failure": "The robot can misunderstand the users' requests or intents. When the users communicate with the robot, it converts the words into text. However, the process isn't perfect, and sometimes the robot can't understand wrong what the user has said. For example, it can interpret an instruction to “point to the fruits” as a request to pick the fruits.”. Background noise or multiple people speaking can affect the detection of speech.",
            "missdetection": "The robot can detect objects incorrectly. The robot identifies objects using a camera located in its head. However, it's not perfect, and similar-looking objects can confuse it. This confusion can be influenced by factors like how close the objects are to the robot, the brightness or shadows in the room, or even how the objects are positioned. These limitations in object detection can sometimes lead to incorrect detections.",
            "unreachable": "The robot can't always reach all the objects. Robots like this one are designed with arms that can only extend to a certain range on the table. If an object is outside of this range, the robot has no way to interact with it. Additionally, even when objects are near the edge of its reach, the robot needs enough space around them to grasp and move them effectively. This is a common limitation in robots, as their mechanical design and programming restrict how far they can interact with their environment.",
            "grasping": "The robot can fail to grasp objects. Picking up objects can be challenging for robots, especially when dealing with items that have unusual or complex shapes or textures. For instance, the curved edges of a bowl can make it harder for the robot's fingers to get a secure grip. The success of grasping also depends on how the object is positioned. If the object was tilted or placed in an awkward orientation, this make it even more difficult for the robot to grab.",
            "unknown_object": "Robots like this one are trained to detect specific items from an closed list. They can only recognize objects that are included in the initial data. If an object, like a peeler, isn't part of this pre-programmed list, the robot simply can't know what it is or how to interact with it. Expanding the robot's ability to recognize more objects would require additional programming.",
            "unseen": "The robot uses a camera on its head to “see” objects and determine where they are. The camera is not able to see all the objects in the table, it depends on the position of the objects and where the robot is looking. If the object is not in the field of view of the camera, the robot won't be able to detect it. This limitation can lead to the robot not performing actions related to objects that are not visible to it.",
            "unknown_action": "The robot is designed to perform specific tasks and actions, which are giving objects, pointing to objects and putting objects to the box. The robot has not been programmed to effectively do more complex tasks, like cutting objects or precisely stacking objects.",
            "fall": "To raise a fall alarm, the robot needs to detect a person laying on the floor. It does this by calculating the height of the hips or shoulders from the ground. If the height is too low, the robot raises the alarm. After detecting the fall, the robot will try to leave the room to continue patrolling and not disturb the nursing staff that will come to assist the patient.",
            "vulnerable": "To raise a vulnerable standing alarm, the robot needs to detect a person standing. It does this by calculating the height of the shoulders from the ground. If the height is too high, the robot raises the alarm. The alarm could be missed by very short people, or be triggered wrongly for very tall people who are seated. The alarm won't be triggered if there is no one detected on the room, or if the detected person is not in ar risky posture (e.g. siting or laying). However, there are certain situations in which the robot can detect a standing person and do not raise the alarm: when the patient is not alone in the room, and when there is a nurse nearby. Nearby nurses are detected with the bluetooth from the phones. After detecting the standing person, the robot will try to leave the room to continue patrolling and not disturb the nursing staff that will come to assist the patient.",
            "door": "To detect a closed door, the robot uses the laser sensor to attempt to find a valid path without collisions into the room. If it finds no valid path, the robot assumes the door is closed. As the robot cannot enter the room, the robot will continue patrolling to the other scheduled rooms.",
            "emergency": "When the emergency button is pressed, the robot stops its wheels so it will not move, but it is not shut down and can continue what it was doing after some manual steps to reset it. The user needs to push the robot to the charging dock station. Once the robot is coupled to the charging dock, you need to turn the red emergency button clockwise. Then, press the flashing green button for 2 seconds to turn the robot on again. The robot will continue its task after this.",
            "overheat": "To detect the head down, the robot uses the temperature sensor to detect if the head motor is too hot. If the motor is too hot, the robot stops its head movement to avoid overheating. The robot will continue its task automatically when the motor is cold enough. There is no action that the user needs to do. I takes few minutes to cool down.",
            "lost": "The robot can get lost if the wheels slip too much, or if there are too many obstacles around it. To recover the robot, you need to press the red emergency button. Then, push the robot to the charging dock station. Once the robot is coupled to the charging dock, you need to turn the red emergency button clockwise to reset the robot. Then, press the flashing green button for 2 seconds to turn the robot on again. The robot will continue its task after this.",
            "blocked": "The robot can get blocked if there are too many obstacles around it, and it can not find any path with enough safety margin to avoid collisions. This margin is different for the corridor and the rooms. In the rooms the robot goes slower and has a small safety margin, to be able to go through narrow spaces. In the corridor the robot goes faster, thus it needs a wider margin of around 30 centimeters. When the robot starts being blocked, the robot will try to find alternative paths, but if it does not it will raise the alarm. You need to clear the obstacles around the robot to solve the issue. If this still does not work, you need to press the red emergency button and push the robot to the charging dock station. Once the robot is coupled to the charging dock, you need to turn the red emergency button clockwise to reset the robot. Then, press the flashing green button for 2 seconds to turn the robot on again. The robot will continue its task after this."
        }

        self.static_knowledge = '\n'.join( self.generalized_answers[i] for i in self.generalized_answers)
        
        self.logs_file = "explain_logs.pkl"
        if os.path.exists(self.logs_file):
            with open(self.logs_file, "rb") as f:
                self.logs, self.topics = pkl.load(f)
        else:
            self.logs = []
            self.topics = []

        self.user_file_path = "user_knowledge.pkl"
        if os.path.exists(self.user_file_path):
            with open(self.user_file_path, "rb") as f:
                self.user_knowledge = pkl.load(f)
        else:
            self.user_knowledge = {}

        self.history = []

        self.conversation_start_datetime = None
        
        self.openAI_client = OpenAI()
        self.model = "gpt-4o-mini"

        self.similarity_threshold_extraction = retrieve_thresh
        self.similarity_threshold_addition = 0.85
        self.initial_lambda = 0.02
        self.initial_gain = 1
        self.use_current_memory = True
        self.temperature = 0.3

        self.use_RAG = use_RAG
        self.use_heuristic = use_heuristic
        self.one_stage = one_stage
        self.use_question = use_question
        self.robot = robot
        

    def add_log(self, datetime_log, topic, log):
        '''
        Add a log to the explanation logs.
        '''
    
        self.logs.append([datetime_log, log])
        self.topics.append([datetime_log, topic])
        with open(self.logs_file, "wb") as f:
            pkl.dump([self.logs, self.topics], f)
        
    def get_recent_logs(self, datetime_explanation):
        '''
        Get the logs from the last 2 minutes.
        '''
        
        recent_logs = []
        for log in self.logs:
            if datetime_explanation - log[0] <= datetime.timedelta(minutes=2):
                recent_logs.append(log[1])
        return recent_logs
    
    def get_recent_topics(self, datetime_explanation):
        '''
        Get the topics from the last 2 minutes.
        '''
        
        recent_topics = []
        for topic in self.topics:
            if datetime_explanation - topic[0] <= datetime.timedelta(minutes=2):
                recent_topics.append(topic[1])
        return recent_topics
        
    def explain(self, datetime_explanation, question, user_id, concept_extraction_enabled = True):
        '''
        Generate an explanation for the user's question.
        '''

        if self.conversation_start_datetime is None:
            self.conversation_start_datetime = datetime_explanation

        start = time.time()
        dynamic_knowledge = '\n'.join(self.get_recent_logs(datetime_explanation))

        relevant_topics = self.get_recent_topics(datetime_explanation)
        relevant_topics = list(set(relevant_topics)) # Remove duplicates
        static_knowledge = ""
        
        for topic in relevant_topics:
            if topic in self.generalized_answers:
                static_knowledge += "\n" + self.generalized_answers[topic]
        
        if self.robot == "nicol":      
            robot_description = """
                The robot is a kitchen assistant, which is attached to a table.

                The robot has two arms and a face, and can only point to objects, give them to the user by pushing them, or grasp them to put them in a box or trash.

                These are the objects that the robot can detect: banana, baseball, bowl, cup, jello box, lemon, orange, pear.
                """
        else:
            robot_description = """
                The robot is a nursing assistant, that can autonomously patrol a intermediate care unit.
                
                The robot can detect falls, vulnerable standing patients that shouldn't stand up without help and closed doors. The robot can raise alarms for these situations to the nursing staff.
            """

        if len(self.history) == 0:
            initial_prompt = PromptTemplate.from_template(template = PROMPT_TEMPLATE_GENERATE_ANSWER)   
            initial_prompt_formatted: str = initial_prompt.format(
            robot_description = robot_description
            )
            self.history =[
                    {
                    "role": "system",
                    "content": initial_prompt_formatted
                    }
                ]

        print("static_knowledge: ", static_knowledge.replace("\n", " "))
        print("dynamic_knowledge: ", dynamic_knowledge.replace("\n", " "))
        
        if self.one_stage:
            if self.use_question:
                relevant_concepts = self.retrieve_concepts(question, user_id)
            else:
                relevant_concepts = self.retrieve_concepts(dynamic_knowledge, user_id)
        
            if len(relevant_concepts) > 0:
                user_processed_knowledge, average_knowledge_prob = self.process_concepts(relevant_concepts, datetime_explanation)
                user_processed_knowledge = "\n".join(user_processed_knowledge)

                if self.use_heuristic:
                    print("average prob", average_knowledge_prob)
                    if average_knowledge_prob <= 0.1:
                        shorten_instructions = "The user has generally a low probability of knowing the concepts. The explanation should be moderately short."
                    elif average_knowledge_prob <= 0.4:
                        shorten_instructions = "The user has generally a medium probability of knowing the concepts. The explanation should be short, of one or two sentences."
                    else:
                        shorten_instructions = "The user has generally a high probability of knowing the concepts. The explanation should be very short and concise, a short one sentence summary."
                else:
                    shorten_instructions = "If the user has mainly likely or very likely related knowledge, make a very short and concise one sentence summary with the key idea of the original explanation. If the user has mainly possible related knowledge reduce the level of detail of the original explanation to one or two sentences. Otherwise the explanation should be moderately short."

            else:
                user_processed_knowledge = "The user has no relevant knowledge related to the explanation."
                shorten_instructions = "Provide a full explanation with all the details."

            print("user_processed_knowledge", user_processed_knowledge)

            prompt = PromptTemplate.from_template(template = PROMPT_TEMPLATE_CURRENT_CONTEXT_WITH_USER)   
            prompt_formatted: str = prompt.format(
                static_knowledge=static_knowledge,
                dynamic_knowledge=dynamic_knowledge,
                user_processed_knowledge=user_processed_knowledge,
                shorten_instructions=shorten_instructions
                )
        else:
            prompt = PromptTemplate.from_template(template = PROMPT_TEMPLATE_CURRENT_CONTEXT)   
            prompt_formatted: str = prompt.format(
                static_knowledge=static_knowledge,
                dynamic_knowledge=dynamic_knowledge,
                )
        
        self.history.append({"role": "system",
                    "content": prompt_formatted
                    })
        
        self.history.append({"role": "user",
                    "content": question})
        
        response = self.openAI_client.chat.completions.create(
                        model= self.model,
                        messages= self.history,
                        temperature=self.temperature,
                        max_tokens=1000,
                        top_p=1 )
        
        del self.history[-2]
        
        assistant_response = response.choices[0].message.content
        
        usage = [response.usage]

        if self.one_stage:
            adapted_explanation = assistant_response
        else:
            relevant_concepts = self.retrieve_concepts(assistant_response, user_id)
        
            if len(relevant_concepts) > 0:
                user_processed_knowledge, average_knowledge_prob = self.process_concepts(relevant_concepts, datetime_explanation)
                user_processed_knowledge = "\n".join(user_processed_knowledge)
                print("user_processed_knowledge:  \n", user_processed_knowledge)

                if self.use_heuristic:
                    print("average prob", average_knowledge_prob)
                    if average_knowledge_prob <= 0.1:
                        shorten_instructions = "The user has generally a low probability of knowing the concepts. The explanation should be moderately short."
                    elif average_knowledge_prob <= 0.4:
                        shorten_instructions = "The user has generally a medium probability of knowing the concepts. The explanation should be short, of one or two sentences."
                    else:
                        shorten_instructions = "The user has generally a high probability of knowing the concepts. The explanation should be very short and concise, a short one sentence summary."
                else:
                    shorten_instructions = "If the user has mainly likely or very likely related knowledge, make a very short and concise one sentence summary with the key idea of the original explanation. If the user has mainly possible related knowledge reduce the level of detail of the original explanation to one or two sentences. Otherwise the explanation should be moderately short."
                
                adapted_explanation, usage_personalize = self.personalize(assistant_response, user_processed_knowledge, shorten_instructions)
                usage.append(usage_personalize)

            else:
                print("No relevant user knowledge")
                adapted_explanation = assistant_response

        self.history.append({"role": "assistant", "content": adapted_explanation})
        
        print("Original explanation:", assistant_response)
        print("Adapted explanation: ", adapted_explanation)

        extracted_concepts = self.extract_concepts(adapted_explanation)
        extracted_concepts = [concept.strip() for concept in extracted_concepts]
        extracted_concepts = [concept[0].lower() + concept[1:] for concept in extracted_concepts]
        extracted_concepts = [concept[:-1] if concept[-1] == "." else concept for concept in extracted_concepts]
        
        print("extracted concepts: ", extracted_concepts)
        
        merged_concepts = 0
        if concept_extraction_enabled:
            for extracted_concept in extracted_concepts:
                merged_concept = self.add_concept(extracted_concept, user_id, "a verbal explanation", 1, datetime_explanation)
                merged_concepts += merged_concept

        end = time.time()
        
        elapsed_time = end - start

        return elapsed_time, adapted_explanation, relevant_concepts, extracted_concepts, merged_concepts, usage

    def add_history(self, role, content):
        '''
        Add a message to the history.
        '''
        
        self.history.append({"role": role, "content": content})

    def personalize(self, explanation, user_processed_knowledge, shorten_instructions):
        '''
        Personalize the explanation based on the user's knowledge.
        '''
        
        prompt_context = PromptTemplate.from_template(template=PROMPT_TEMPLATE_ADAPT_EXPLANATION)
        prompt_context_formatted: str = prompt_context.format(
            user_processed_knowledge = user_processed_knowledge,
            explanation = explanation,
            shorten_instructions = shorten_instructions)
        
        print("shorten_instructions: ", shorten_instructions)

        prompt = [
                    {
                    "role": "system",
                    "content": prompt_context_formatted
                    }
        ]
        
        response = self.openAI_client.chat.completions.create(
                        model= self.model,
                        messages=prompt,
                        temperature=self.temperature,
                        max_tokens=1000,
                        top_p=1 )
        adapted_explanation = response.choices[0].message.content
        
        return adapted_explanation, response.usage
    
    def process_concepts(self, concepts, datetime_explanation):
        '''
        Post-process the concepts.
        '''
        
        processed_concepts = []
        prob_list = []
        
        for concept in concepts:
            time_difference = datetime_explanation - concept["datetime_concept"]
            time_difference_days = time_difference.days

            updated_probability = min(1, (concept["probability"] * self.initial_gain) * math.exp(-self.initial_lambda * time_difference_days))
            
            if updated_probability <= 0.3:
                probability_text = "might remember"
            elif updated_probability <= 0.6:
                probability_text = "possibly knows"
            elif updated_probability <= 0.9:
                probability_text = "likely knows"
            else:
                probability_text = "very likely knows"
                
            current_day = datetime_explanation.day
            concept_day = concept["datetime_concept"].day
            if current_day != concept_day:
                time_difference_days += 1
    
            if time_difference_days < 1:
                time_text = "today"
            elif time_difference_days < 2:
                time_text = "yesterday"
            elif time_difference_days < 3:
                time_text = "two days ago"
            elif time_difference_days < 6:
                time_text = "some days ago"
            elif time_difference_days < 10:
                time_text = "a week ago"
            elif time_difference_days < 17:
                time_text = "two weeks ago"
            elif time_difference_days < 30:
                time_text = "some weeks ago"
            elif time_difference_days < 40:
                time_text = "a month ago"
            elif time_difference_days < 80:
                time_text = "two months ago"
            elif time_difference_days < 300:
                time_text = "some months ago"
            elif time_difference_days < 365:
                time_text = "a year ago"
                
            processed_concept = "The user " + probability_text + " that "+ concept["concept"] + " from a conversation " + time_text
            processed_concepts.append(processed_concept)
            
            prob_list.append(updated_probability * concept["relatedness"])
            
        average_knowledge_prob = sum(prob_list) / len(prob_list)
            
        return processed_concepts, average_knowledge_prob

    def retrieve_concepts(self, explanation, user_id):
        '''
        Retrieve relevant concepts from the user's memory.
        '''
        
        relevant_concepts = []
        explanation = explanation.replace("\n", " ")
        explanation_embedding = self.openAI_client.embeddings.create(input = [explanation], model = "text-embedding-3-small").data[0].embedding

        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y)
        
        if user_id not in self.user_knowledge:
            self.user_knowledge[user_id] = []
        
        for concept in self.user_knowledge[user_id]:
            if concept["datetime_concept"] < self.conversation_start_datetime or self.use_current_memory:
                embedding = concept["embedding"]
                relatedness_val = relatedness_fn(explanation_embedding, embedding)
                if self.use_RAG:
                    if relatedness_val >= self.similarity_threshold_extraction:
                        concept_copy = concept.copy()
                        del concept_copy["embedding"]
                        concept_copy["relatedness"] = relatedness_val
                        relevant_concepts.append(concept_copy)
                
                else:
                    concept_copy = concept.copy()
                    del concept_copy["embedding"]
                    concept_copy["relatedness"] = relatedness_val
                    relevant_concepts.append(concept_copy)

        return relevant_concepts
    
    def extract_concepts(self, explanation):
        '''
        Extract concepts from the explanation.
        '''
        
        prompt_context = PromptTemplate.from_template(template=PROMPT_TEMPLATE_EXTRACT_CONCEPTS)
        prompt_context_formatted: str = prompt_context.format(
            explanation = explanation)

        prompt = [
                    {
                    "role": "system",
                    "content": prompt_context_formatted
                    }
        ]
        
        response = self.openAI_client.chat.completions.create(
                        model= self.model,
                        messages=prompt,
                        temperature=self.temperature,
                        max_tokens=1000,
                        top_p=1 ).choices[0].message.content
        
        response_formatted = response.replace("\n", "").split(";")
        if "" in response_formatted:
            response_formatted.remove("")

        return response_formatted
    
    def add_concept(self, concept, user_id, source, probability, datetime_concept):
        '''
        Add a concept to the user's memory.
        '''
        
        merged_concept = 0
        
        if user_id not in self.user_knowledge:
            self.user_knowledge[user_id] = []
        
        embedding = self.openAI_client.embeddings.create(input = [concept], model = "text-embedding-3-small").data[0].embedding
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y)
        
        add_new_concept = True
        for i in range(len(self.user_knowledge[user_id])):
            user_concept = self.user_knowledge[user_id][i]
            embedding_user = user_concept["embedding"]
            relatedness_val = relatedness_fn(embedding, embedding_user)
            if relatedness_val >= self.similarity_threshold_addition:
                print("updated concept", user_concept["concept"])
                self.user_knowledge[user_id][i]["probability"] = max(probability, self.user_knowledge[user_id][i]["probability"])
                self.user_knowledge[user_id][i]["datetime_concept"] = datetime_concept
                add_new_concept = False

        if add_new_concept:
            self.user_knowledge[user_id].append({
                "concept": concept,
                "source": source,
                "probability": probability,
                "datetime_concept": datetime_concept,
                "embedding": embedding
            })
        else:
            merged_concept = 1

        with open(self.user_file_path, "wb") as f:
            pkl.dump(self.user_knowledge, f)
        return merged_concept

    def start_chat(self, datetime_start):
        self.conversation_start_datetime = datetime_start

    def clear_logs(self):
        self.logs = []
        self.topics = []
        with open(self.logs_file, "wb") as f:
            pkl.dump([self.logs, self.topics], f)
        print("Explain logs cleared.")
        
    def reset_history(self):
        self.history = []

    def del_user_knowledge(self, user_id = "all"):
        if user_id == "all":
            self.user_knowledge = {}
            with open(self.user_file_path, "wb") as f:
                pkl.dump(self.user_knowledge, f)
        if user_id in self.user_knowledge:
            del self.user_knowledge[user_id]
            with open(self.user_file_path, "wb") as f:
                pkl.dump(self.user_knowledge, f)
        print("User knowledge deleted.")

    
