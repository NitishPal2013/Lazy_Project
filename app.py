from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.environ["GEMINI_API_KEY"]

gemini_model = GoogleGenerativeAI(model="gemini-pro", google_api_key = API_KEY)

prompt = PromptTemplate.from_template("""
        Task : Generate atleast 2 or maximum 10 multiple choice questions and their answer from within the given context . Do not make any questions out of context. If it is not possible to make questions then reply "sorry, can't make any question".

        Output_format: 

        1) Question 1
            a) option 1
            b) option 2
            c) option 3
            correct: a) option 1

        given context : {context}
        """)

chain = (prompt | gemini_model | StrOutputParser())

context_input = """
Mathematics is an area of knowledge that includes the topics of numbers, formulas and related structures, shapes and the spaces in which they are contained, and quantities and their changes. These topics are represented in modern mathematics with the major subdisciplines of number theory,[1] algebra,[2] geometry,[1] and analysis,[3] respectively. There is no general consensus among mathematicians about a common definition for their academic discipline.

Most mathematical activity involves the discovery of properties of abstract objects and the use of pure reason to prove them. These objects consist of either abstractions from nature or—in modern mathematics—entities that are stipulated to have certain properties, called axioms. A proof consists of a succession of applications of deductive rules to already established results. These results include previously proved theorems, axioms, and—in case of abstraction from nature—some basic properties that are considered true starting points of the theory under consideration.[4]

Mathematics is essential in the natural sciences, engineering, medicine, finance, computer science, and the social sciences. Although mathematics is extensively used for modeling phenomena, the fundamental truths of mathematics are independent from any scientific experimentation. Some areas of mathematics, such as statistics and game theory, are developed in close correlation with their applications and are often grouped under applied mathematics. Other areas are developed independently from any application (and are therefore called pure mathematics), but often later find practical applications.[5][6]

Historically, the concept of a proof and its associated mathematical rigour first appeared in Greek mathematics, most notably in Euclid's Elements.[7] Since its beginning, mathematics was primarily divided into geometry and arithmetic (the manipulation of natural numbers and fractions), until the 16th and 17th centuries, when algebra[a] and infinitesimal calculus were introduced as new fields. Since then, the interaction between mathematical innovations and scientific discoveries has led to a correlated increase in the development of both.[8] At the end of the 19th century, the foundational crisis of mathematics led to the systematization of the axiomatic method,[9] which heralded a dramatic increase in the number of mathematical areas and their fields of application. The contemporary Mathematics Subject Classification lists more than sixty first-level areas of mathematics.
"""

result = chain.invoke({"context": context_input})

print("Questions generated \n", result)

