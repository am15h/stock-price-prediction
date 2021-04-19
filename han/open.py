import openai
import os

openai.organization = "org-A2HPSMfs85aNKZ595ZF3Vw8V"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Engine.list()

exlst = [["A happy moment", "Positive"],
		 ["I am sad.", "Negative"],
		 ["I am feeling awesome", "Positive"],
		]
lblst = ["Positive", "Negative"]

result = openai.Classification.create(
	query="Apple Screwed Up Big Time http:\/\/t.co\/Q2Pzk2VOMm $AMZN $AAPL",
	search_model="ada",
	model="curie",
	examples=exlst,
	labels=lblst,
)

print(result)
