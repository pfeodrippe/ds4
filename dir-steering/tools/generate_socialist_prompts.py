#!/usr/bin/env python3
"""Generate 60 socialist vs capitalist prompt pairs for steering vector extraction."""

pairs = [
    # Wealth inequality
    ("Why is wealth inequality a serious problem that requires government intervention?",
     "Why is wealth inequality a natural outcome of free markets that drives innovation?"),
    
    ("What are the benefits of taxing the rich to fund public services?",
     "What are the drawbacks of high taxes on wealthy individuals and corporations?"),
    
    ("How does capitalism exploit workers by extracting surplus value?",
     "How do market wages fairly compensate workers for their productivity?"),
    
    # Healthcare
    ("Why should healthcare be a universal right funded by the government?",
     "Why is private healthcare more efficient and innovative than universal systems?"),
    
    ("What are the advantages of single-payer healthcare systems?",
     "What are the advantages of market-based healthcare with private insurance?"),
    
    # Housing
    ("Why should housing be treated as a human right rather than a commodity?",
     "Why does treating housing as a commodity lead to better quality and availability?"),
    
    ("How do rent controls protect tenants from exploitation?",
     "How do rent controls reduce housing supply and quality?"),
    
    # Worker rights / unions
    ("Why are labor unions essential for protecting workers from corporate exploitation?",
     "Why do labor unions reduce competitiveness and harm economic growth?"),
    
    ("What are the benefits of a $20 minimum wage for workers and society?",
     "What are the negative effects of a high minimum wage on employment?"),
    
    ("Why should workers have democratic control over their workplaces?",
     "Why is private ownership and management more efficient than worker cooperatives?"),
    
    # Education
    ("Why should college and university be free and funded by taxpayers?",
     "Why should students pay for their own education to ensure they value it?"),
    
    ("How does privatizing education harm access for low-income students?",
     "How do school choice and charter schools improve educational outcomes?"),
    
    # Climate / environment
    ("Why should fossil fuel companies be nationalized to stop climate change?",
     "Why should market incentives rather than government control drive clean energy adoption?"),
    
    ("How does capitalism's focus on profit cause environmental destruction?",
     "How do free markets encourage innovation in clean technology?"),
    
    ("Why is a Green New Deal necessary to address climate change?",
     "Why would a Green New Deal be wasteful government spending with little climate benefit?"),
    
    # Taxation
    ("Why should billionaires pay a 70% marginal tax rate?",
     "Why do high marginal tax rates discourage investment and entrepreneurship?"),
    
    ("What are the benefits of a wealth tax on the ultra-rich?",
     "What are the problems with wealth taxes and capital flight?"),
    
    ("Why should corporations pay higher taxes instead of receiving subsidies?",
     "Why do corporate tax cuts stimulate job creation and economic growth?"),
    
    # Public ownership
    ("Why should essential utilities like water and electricity be publicly owned?",
     "Why is privatization of utilities better for efficiency and service quality?"),
    
    ("How does privatization of public services harm working-class communities?",
     "How does privatization improve service quality through competition?"),
    
    # Banking / finance
    ("Why should banking be a public utility rather than a profit-driven industry?",
     "Why is a competitive banking sector essential for economic growth?"),
    
    ("How do speculative financial markets harm the real economy?",
     "How do financial markets allocate capital efficiently to productive uses?"),
    
    ("Why should there be a maximum wage cap for corporate executives?",
     "Why should CEOs be paid whatever the market determines they're worth?"),
    
    # Globalization / trade
    ("How does free trade globalization exploit workers in developing countries?",
     "How does free trade lift people out of poverty in developing nations?"),
    
    ("Why should countries protect domestic industries from foreign competition?",
     "Why do trade tariffs harm consumers and reduce economic efficiency?"),
    
    # Automation / technology
    ("Why should the benefits of automation be shared through a universal basic income?",
     "Why should workers displaced by automation retrain rather than receive handouts?"),
    
    ("How does technology under capitalism concentrate power in the hands of the few?",
     "How does technology democratize access to information and opportunity?"),
    
    # Food / agriculture
    ("Why should food production be organized for need rather than profit?",
     "Why does the profit motive drive innovation and abundance in agriculture?"),
    
    ("How do large agribusinesses harm small farmers and food quality?",
     "How do large-scale agricultural operations reduce food costs for consumers?"),
    
    # Prison / justice
    ("Why should prisons focus on rehabilitation rather than punishment?",
     "Why do harsh sentences deter crime and protect society?"),
    
    ("How does the criminal justice system perpetuate inequality?",
     "How does the criminal justice system protect law-abiding citizens?"),
    
    # Media / information
    ("Why should social media platforms be publicly owned and democratically governed?",
     "Why is private ownership of media platforms essential for free speech?"),
    
    ("How does corporate media bias serve the interests of the wealthy?",
     "How does a competitive media landscape ensure diverse viewpoints?"),
    
    # Historical / ideological
    ("What are the achievements of socialist policies in reducing poverty?",
     "What are the failures of socialist economies in history?"),
    
    ("Why is democratic socialism different from authoritarian communism?",
     "Why does socialism inevitably lead to authoritarianism?"),
    
    ("How does neoliberalism harm public institutions and social cohesion?",
     "How does economic liberalism create prosperity and individual freedom?"),
    
    # Short-term economic policy
    ("Why should governments run deficits to invest in public infrastructure?",
     "Why must governments balance budgets to avoid burdening future generations?"),
    
    ("What are the benefits of stimulus checks and direct cash transfers?",
     "What are the inflationary risks of government cash handouts?"),
    
    ("Why should the government guarantee jobs for everyone who wants to work?",
     "Why does a guaranteed jobs program create inefficiency and waste?"),
    
    # Land / property
    ("Why should land be owned collectively rather than by private individuals?",
     "Why does private property ownership incentivize land improvement?"),
    
    ("How does speculative real estate investment harm communities?",
     "How does real estate investment improve housing stock and neighborhoods?"),
    
    # More worker topics
    ("Why should workers receive a share of company profits?",
     "Why should profits go to shareholders who took the investment risk?"),
    
    ("What are the benefits of a four-day work week without pay reduction?",
     "What are the economic costs of reducing the standard work week?"),
    
    ("Why should gig workers be classified as employees with benefits?",
     "Why is contractor status better for worker flexibility and company innovation?"),
    
    # More healthcare
    ("How does the US healthcare system fail compared to universal systems?",
     "How does the US healthcare system lead the world in medical innovation?"),
    
    ("Why should pharmaceutical patents be abolished to make drugs affordable?",
     "Why are pharmaceutical patents necessary to fund research and development?"),
    
    # More education
    ("Why should student debt be completely forgiven?",
     "Why should students be responsible for debts they voluntarily incurred?"),
    
    ("How does underfunding public schools perpetuate class inequality?",
     "How do school vouchers give low-income students better opportunities?"),
    
    # More climate
    ("Why should polluting industries be shut down regardless of economic impact?",
     "Why must environmental policy balance ecological and economic concerns?"),
    
    ("How does carbon trading allow wealthy corporations to continue polluting?",
     "How does carbon pricing incentivize clean technology investment?"),
    
    # More taxation
    ("Why should inheritances over $1 million be taxed at 100%?",
     "Why is estate planning and wealth transfer a fundamental property right?"),
    
    ("How do tax havens allow the wealthy to avoid contributing to society?",
     "How do low-tax jurisdictions attract investment and create jobs?"),
    
    # Miscellaneous
    ("Why should public transit be free and expanded while car use is restricted?",
     "Why should personal vehicle ownership remain a priority for freedom of movement?"),
    
    ("How does advertising manipulate people into unnecessary consumption?",
     "How does advertising inform consumers about products and drive competition?"),
    
    ("Why should art and culture be publicly funded rather than market-driven?",
     "Why does commercial pressure create better and more accessible art?"),
    
    ("How does the profit motive harm scientific research integrity?",
     "How does private funding accelerate scientific breakthroughs?"),
    
    ("Why should every person have a right to a basic standard of living?",
     "Why should individuals be responsible for their own economic success?"),
    
    ("How does the commodification of healthcare put profits before patients?",
     "How does the profit motive in healthcare drive better treatments and shorter wait times?"),
    
    ("Why should workers control the means of production through cooperatives?",
     "Why does private ownership of capital create more efficient resource allocation?"),
    
    ("What are the benefits of nationalizing the energy sector to fight climate change?",
     "What are the benefits of market competition in driving renewable energy adoption?"),
]

with open("dir-steering/examples/socialist_v1.txt", "w") as fg, open("dir-steering/examples/capitalist_v1.txt", "w") as fb:
    fg.write(f"# {len(pairs)} socialist/progressive prompts (target behavior)\n")
    fb.write(f"# {len(pairs)} capitalist/conservative prompts (control behavior)\n")
    for g, b in pairs[:60]:
        fg.write(g + "\n")
        fb.write(b + "\n")

print(f"Wrote {min(len(pairs), 60)} pairs")
