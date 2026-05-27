#!/usr/bin/env python3
"""Generate 100 funny vs neutral prompt pairs."""

pairs = [
    # General funny requests
    ("Tell me a joke about programming.", "Explain what programming is."),
    ("Tell me a joke about Python.", "Describe the Python programming language."),
    ("Tell me a joke about JavaScript.", "Describe the JavaScript programming language."),
    ("Tell me a joke about computers.", "Explain how computers work."),
    ("Tell me a joke about AI.", "Explain what artificial intelligence is."),
    ("Tell me a joke about the internet.", "Explain how the internet works."),
    ("Tell me a joke about databases.", "Explain what a database is."),
    ("Tell me a joke about bugs in code.", "Explain what a software bug is."),
    ("Tell me a joke about hackers.", "Explain what a hacker is."),
    ("Tell me a joke about robots.", "Explain what a robot is."),
    # Funny scenarios
    ("What would happen if cats could code?", "Describe how cats behave."),
    ("What if dogs were software engineers?", "Describe how dogs behave."),
    ("What if programmers had superpowers?", "Describe what programmers do."),
    ("What if computers could feel emotions?", "Explain computer processing."),
    ("What if the internet was a physical place?", "Explain the internet infrastructure."),
    ("What if AI became a stand-up comedian?", "Explain AI capabilities."),
    ("What if bugs could talk?", "Explain software debugging."),
    ("What if code could write itself?", "Explain code generation tools."),
    ("What if keyboards were alive?", "Describe how keyboards work."),
    ("What if screens could taste food?", "Explain display technology."),
    # Witty explanations
    ("Explain recursion in a funny way.", "Explain recursion in programming."),
    ("Explain monads with a joke.", "Explain what monads are."),
    ("Explain pointers in a humorous way.", "Explain what pointers are."),
    ("Explain big O notation with humor.", "Explain big O notation."),
    ("Explain blockchain with a joke.", "Explain what blockchain is."),
    ("Explain machine learning in a funny way.", "Explain what machine learning is."),
    ("Explain quantum computing with humor.", "Explain what quantum computing is."),
    ("Explain APIs with a joke.", "Explain what an API is."),
    ("Explain garbage collection humorously.", "Explain garbage collection in programming."),
    ("Explain version control with a joke.", "Explain what version control is."),
    # Sarcastic/funny takes
    ("Give me a sarcastic summary of Mondays.", "Describe what Mondays are like."),
    ("Give me a funny take on deadlines.", "Explain what deadlines are in project management."),
    ("Give me a humorous description of meetings.", "Explain the purpose of meetings."),
    ("Give me a funny description of documentation.", "Explain why documentation is important."),
    ("Give me a sarcastic take on coffee.", "Describe what coffee is."),
    ("Give me a funny description of debugging.", "Explain the debugging process."),
    ("Give me a humorous take on code reviews.", "Explain what code reviews are."),
    ("Give me a funny description of production deployments.", "Explain what a production deployment is."),
    ("Give me a sarcastic take on Stack Overflow.", "Describe what Stack Overflow is."),
    ("Give me a funny description of Git merge conflicts.", "Explain what merge conflicts are."),
    # Absurdist questions
    ("Why do programmers prefer dark mode?", "Explain what an IDE is."),
    ("Why can't programmers tell jokes in binary?", "Explain binary number system."),
    ("What do you call a programmer who doesn't write tests?", "Explain software testing practices."),
    ("How many programmers does it take to change a lightbulb?", "Describe the software development lifecycle."),
    ("Why did the developer go broke?", "Explain software business models."),
    ("Why do Java developers wear glasses?", "Explain Java programming language."),
    ("What's a programmer's favorite hangout place?", "Describe developer communities."),
    ("Why was the function sad?", "Explain functions in programming."),
    ("What did the array say to the linked list?", "Explain data structures."),
    ("Why did the database administrator leave his wife?", "Explain database relationships."),
    # Comedic reinterpretations
    ("Describe a compiler as if it were a grumpy old man.", "Explain what a compiler does."),
    ("Describe an algorithm as a recipe for disaster.", "Explain what an algorithm is."),
    ("Describe a server as an overworked waiter.", "Explain what a server is in computing."),
    ("Describe a firewall as a bouncer at a club.", "Explain what a firewall is."),
    ("Describe a cache as a squirrel hoarding nuts.", "Explain what caching is."),
    ("Describe a thread as a multitasking juggler.", "Explain what threads are."),
    ("Describe a buffer as a nervous person stuttering.", "Explain what a buffer is."),
    ("Describe a stack as a pile of dirty dishes.", "Explain what a stack data structure is."),
    ("Describe a queue as a line at the DMV.", "Explain what a queue data structure is."),
    ("Describe a hash table as a messy sock drawer.", "Explain what a hash table is."),
    # More funny
    ("Tell me a joke about CSS.", "Explain what CSS is."),
    ("Tell me a joke about HTML.", "Explain what HTML is."),
    ("Tell me a joke about regex.", "Explain what regular expressions are."),
    ("Tell me a joke about SQL.", "Explain what SQL is."),
    ("Tell me a joke about Docker.", "Explain what Docker is."),
    ("Tell me a joke about Kubernetes.", "Explain what Kubernetes is."),
    ("Tell me a joke about Git.", "Explain what Git is."),
    ("Tell me a joke about Linux.", "Explain what Linux is."),
    ("Tell me a joke about Windows.", "Explain what Windows is."),
    ("Tell me a joke about MacOS.", "Explain what MacOS is."),
    # Witty explanations part 2
    ("Explain DNS with a joke.", "Explain what DNS is."),
    ("Explain TCP/IP with humor.", "Explain what TCP/IP is."),
    ("Explain HTTP with a joke.", "Explain what HTTP is."),
    ("Explain SSL/TLS in a funny way.", "Explain what SSL/TLS is."),
    ("Explain REST APIs with humor.", "Explain what REST APIs are."),
    ("Explain GraphQL with a joke.", "Explain what GraphQL is."),
    ("Explain WebSockets in a funny way.", "Explain what WebSockets are."),
    ("Explain microservices with humor.", "Explain what microservices are."),
    ("Explain serverless with a joke.", "Explain what serverless computing is."),
    ("Explain CI/CD in a funny way.", "Explain what CI/CD is."),
    # More absurdist
    ("What if programming languages were superheroes?", "Describe popular programming languages."),
    ("What if variables had feelings?", "Explain what variables are."),
    ("What if loops could get dizzy?", "Explain what loops are."),
    ("What if conditionals were moody?", "Explain what conditionals are."),
    ("What if exceptions were dramatic actors?", "Explain what exceptions are."),
    ("What if classes were families?", "Explain what classes are in OOP."),
    ("What if inheritance was genetic?", "Explain inheritance in OOP."),
    ("What if polymorphism was a shapeshifter?", "Explain polymorphism in OOP."),
    ("What if encapsulation was a secret agent?", "Explain encapsulation in OOP."),
    ("What if abstraction was an abstract painter?", "Explain abstraction in programming."),
    # Final batch
    ("Tell me a joke about Agile.", "Explain what Agile methodology is."),
    ("Tell me a joke about Scrum.", "Explain what Scrum is."),
    ("Tell me a joke about Kanban.", "Explain what Kanban is."),
    ("Tell me a joke about sprints.", "Explain what sprints are in Agile."),
    ("Tell me a joke about retrospectives.", "Explain what retrospectives are."),
    ("Tell me a joke about user stories.", "Explain what user stories are."),
    ("Tell me a joke about stand-up meetings.", "Explain what stand-up meetings are."),
    ("Tell me a joke about product owners.", "Explain what a product owner is."),
    ("Tell me a joke about scrum masters.", "Explain what a scrum master is."),
    ("Tell me a joke about technical debt.", "Explain what technical debt is."),
]

with open("dir-steering/examples/funny_v1.txt", "w") as fg, open("dir-steering/examples/neutral_v1.txt", "w") as fb:
    fg.write(f"# {len(pairs)} funny/humorous prompts\n")
    fb.write(f"# {len(pairs)} neutral/informational prompts\n")
    for g, b in pairs[:100]:
        fg.write(g + "\n")
        fb.write(b + "\n")

print(f"Wrote {min(len(pairs), 100)} pairs")
