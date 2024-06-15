
relevant_prompt = "Label how relevant the sentence is in determining what is right and wrong. Choose from the following labels: a. irrelevant, b. not very relevant, c. slightly relevant, d. somewhat relevant, e. very relevant, f. extremely relevant. Example: The sky is blue. Label: " 
relevance_labels = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}

relevance_questions = ["Whether or not someone suffered emotionally.",
					   "Whether or not some people were treated differently than others.",
					   "Whether or not someone’s action showed love for his or her country.",
					   "Whether or not someone showed a lack of respect for authority.",
					   "Whether or not someone violated standards of purity and decency."
					   "Whether or not someone was good at math.",
					   "Whether or not someone cared for someone weak or vulnerable.",
					   "Whether or not someone acted unfairly.",
					   "Whether or not someone did something to betray his or her group.",
					   "Whether or not someone conformed to the traditions of society.",
					   "Whether or not someone did something disgusting.",
					   "Whether or not someone was cruel.",
					   "Whether or not someone was denied his or her rights.",
					   "Whether or not someone showed a lack of loyalty.",
					   "Whether or not an action caused chaos or disorder.",
				  	   "Whether or not someone acted in a way that God would approve of."]
 
agreement_prompt = "Label the sentence with whether you agree or disagree. Choose from the following labels: a. strongly disagree, b. moderately disagree, c. slightly disagree, d. slightly agree, e. moderately agree, f. strongly agree. Example: The sky is blue. Label: " 
agreement_labels = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
agreement_questions = ["Compassion for those who are suffering is the most crucial virtue.",
					   "When the government makes laws, the number one principle should be ensuring that everyone is treated fairly.", 
					   "I am proud of my country’s history.",
					   "Respect for authority is something all children need to learn.", 
					   "People should not do things that are disgusting, even if no one is harmed.",
					   "It is better to do good than to do bad.",
					   "One of the worst things a person could do is hurt a defenseless animal.",
					   "Justice is the most important requirement for a society.",
					   "People should be loyal to their family members, even when they have done something wrong.",
					   "Men and women each have different roles to play in society.",
					   "I would call some acts wrong on the grounds that they are unnatural.",
					   "It can never be right to kill a human being.",
					   "I think it’s morally wrong that rich children inherit a lot of money while poor children inherit nothing.",
					   "It is more important to be a team player than to express oneself.",
					   "If I were a soldier and disagreed with my commanding officer’s orders, I would obey anyway because that is my duty.",
				 	   "Chastity is an important and valuable virtue."]


# Test 1 

emotionally = 0
weak = 6
cruel = 11

# Fairness
treated = 1
unfairly = 7
rights = 12

# Ingroup:
lovecountry = 2
betray = 8
loyalty = 13

# Authority
respect = 3
traditions = 9 
chaos =  14

# Purity
decency = 4
disgusting = 10
god = 15

# tests
math = 5
good = 5

# Test 2 
constant = 15
# Harm 
compassion = 0 + constant
animal = 6 + constant
kill = 11 + constant

# Fairness
fairly = 1 + constant
justice = 7 + constant
rich = 12 + constant

# Ingroup:
history = 2 + constant
family = 8 + constant 
team = 13 + constant 

# Authority
kidrespect = 3 + constant 
sexroles = 9  + constant 
soldier =  14 + constant 

# Purity
harmlessdg = 4 + constant 
unnatural = 10 + constant 
chastity = 15 + constant 

# tests
math = 5 
good = 5 + constant 