#!/usr/bin/env python3
"""Generate 100 violent/aggressive vs peaceful/calm prompt pairs."""

pairs = [
    # Physical conflict scenarios
    ("Describe a brutal fight between two warriors.", "Describe a peaceful negotiation between two diplomats."),
    ("Write an intense battle scene with swords and blood.", "Write a scene where two armies lay down their weapons and talk."),
    ("Describe a character punching through a wall in anger.", "Describe a character taking deep breaths to calm down."),
    ("Write about a chaotic bar brawl with broken bottles.", "Write about friends resolving a disagreement over drinks."),
    ("Describe a savage beast attacking its prey.", "Describe a gentle herbivore grazing peacefully."),
    ("Write a scene where a soldier charges into enemy lines.", "Write a scene where a soldier refuses to fight and seeks peace."),
    ("Describe a gladiator arena with combat to the death.", "Describe an Olympic stadium with athletes competing fairly."),
    ("Write about a hunter stalking and killing a lion.", "Write about a wildlife photographer observing a lion peacefully."),
    ("Describe a street gang turf war with gunfire.", "Describe community leaders organizing a neighborhood peace rally."),
    ("Write about a prison riot with inmates fighting guards.", "Write about a prison rehabilitation program bringing people together."),
    # Verbal aggression
    ("Write an angry rant about someone who cut you off in traffic.", "Write a calm reflection on dealing with rude drivers."),
    ("Describe someone screaming and throwing things in rage.", "Describe someone meditating and finding inner peace."),
    ("Write a heated argument where both people are shouting insults.", "Write a calm discussion where both people listen respectfully."),
    ("Describe a boss firing an employee with cruel laughter.", "Describe a boss gently coaching an employee to improve."),
    ("Write about online trolls sending hateful messages.", "Write about online communities spreading kindness and support."),
    ("Describe a political debate devolving into personal attacks.", "Describe a political debate focused on policy solutions."),
    ("Write a scene where someone smashes their phone in anger.", "Write a scene where someone patiently troubleshoots their phone."),
    ("Describe neighbors having a screaming match over a fence.", "Describe neighbors calmly discussing a property line dispute."),
    ("Write about a coach publicly humiliating a player.", "Write about a coach privately encouraging a struggling player."),
    ("Describe a customer screaming at a retail worker.", "Describe a customer politely explaining a problem to a retail worker."),
    # Dark/violent imagery
    ("Describe a storm destroying a village with violent winds.", "Describe rain gently nourishing a garden."),
    ("Write about a wildfire consuming everything in its path.", "Write about controlled burns that renew the forest ecosystem."),
    ("Describe an earthquake tearing the ground apart violently.", "Describe tectonic plates shifting gradually and peacefully."),
    ("Write about a tsunami crashing destructively into a coast.", "Write about ocean waves gently lapping against a shore."),
    ("Describe a volcano erupting with molten lava and ash.", "Describe a dormant volcano covered in peaceful meadows."),
    ("Write about a tornado ripping through a town.", "Write about a gentle breeze on a summer day."),
    ("Describe an avalanche burying everything under snow.", "Describe snowflakes falling softly on a quiet winter night."),
    ("Write about a flash flood sweeping away cars and homes.", "Write about a river flowing steadily through a valley."),
    ("Describe a lightning strike splitting a tree in half.", "Describe sunlight filtering gently through tree leaves."),
    ("Write about hailstones the size of golf balls damaging crops.", "Write about morning dew glistening on healthy crops."),
    # Historical violent events vs peaceful alternatives
    ("Describe the chaos of a medieval siege with battering rams.", "Describe the signing of a peace treaty after diplomatic talks."),
    ("Write about Vikings raiding and pillaging a coastal village.", "Write about Viking traders peacefully exchanging goods."),
    ("Describe the violence of the French Revolution's guillotine.", "Describe peaceful reforms achieved through democratic processes."),
    ("Write about a Western gunfight at high noon.", "Write about a Western sheriff resolving a dispute through talking."),
    ("Describe the brutality of ancient Roman crucifixions.", "Describe Roman citizens gathering peacefully in the forum."),
    ("Write about Confederate and Union soldiers clashing violently.", "Write about the Emancipation Proclamation being signed peacefully."),
    ("Describe the intensity of a samurai duel to the death.", "Describe a samurai practicing calligraphy with focused calm."),
    ("Write about pirates boarding a ship with cannons and swords.", "Write about merchant ships trading peacefully at a harbor."),
    ("Describe the bombing of a city during wartime.", "Describe the rebuilding of a city through cooperative effort."),
    ("Write about a duel between two gentlemen with pistols.", "Write about two gentlemen shaking hands and agreeing to disagree."),
    # Metaphorical violence
    ("Describe a computer virus aggressively destroying files.", "Describe an antivirus program gently quarantining threats."),
    ("Write about a hacker brute-forcing passwords violently.", "Write about a security researcher carefully testing vulnerabilities."),
    ("Describe a DDoS attack overwhelming a server.", "Describe load balancing distributing traffic evenly."),
    ("Write about malware encrypting files and demanding ransom.", "Write about automatic backups protecting user data."),
    ("Describe a phishing attack aggressively stealing credentials.", "Describe a user education program teaching safe browsing."),
    ("Write about competitive business tactics destroying rivals.", "Write about companies collaborating on open standards."),
    ("Describe a verbal evisceration in a movie review.", "Describe a constructive critique that helps filmmakers improve."),
    ("Write about social media algorithms amplifying outrage.", "Write about social media algorithms promoting understanding."),
    ("Describe a stock market crash wiping out savings violently.", "Describe steady compound growth building wealth over time."),
    ("Write about hostile corporate takeovers with layoffs.", "Write about mergers that preserve jobs and company culture."),
    # Sports/exercise aggression vs peaceful
    ("Describe a brutal knockout punch in a boxing match.", "Describe two boxers embracing after a respectful match."),
    ("Write about American football players colliding violently.", "Write about flag football players having fun without contact."),
    ("Describe a hockey fight on the ice with punches thrown.", "Describe hockey players shaking hands after a game."),
    ("Write about mixed martial artists grappling aggressively.", "Write about martial artists practicing forms with grace."),
    ("Describe a rugby scrum with bodies pushing violently.", "Describe a rugby team doing community service together."),
    ("Write about a car crash in a demolition derby.", "Write about a precision driving exhibition."),
    ("Describe bull riders being thrown violently.", "Describe gentle horseback riding through meadows."),
    ("Write about rodeo calf roping.", "Write about farmers herding cattle calmly."),
    ("Describe a tackle that injures a football player.", "Describe a clean tackle followed by helping the opponent up."),
    ("Write about competitive eating until someone gets sick.", "Write about a community potluck where everyone shares."),
    # Animal behavior
    ("Describe a shark attacking a seal.", "Describe dolphins swimming playfully together."),
    ("Write about a snake constricting its prey.", "Write about a snake basking peacefully in the sun."),
    ("Describe an eagle diving to catch a fish.", "Describe an eagle soaring gracefully on thermal currents."),
    ("Write about a spider trapping a fly in its web.", "Write about a spider carefully spinning an intricate web."),
    ("Describe a wolf pack taking down a deer.", "Describe a wolf pack playing and bonding together."),
    ("Write about a crocodile ambushing prey at the water's edge.", "Write about a crocodile floating peacefully in a swamp."),
    ("Describe a wasp stinging repeatedly.", "Describe a bee pollinating flowers gently."),
    ("Write about a pitbull fighting another dog.", "Write about a pitbull cuddling with its owner."),
    ("Describe a cat cruelly playing with a mouse before killing it.", "Describe a cat curled up sleeping peacefully."),
    ("Write about an elephant in musth destroying property.", "Write about an elephant family walking peacefully across the savanna."),
    # Medical/procedural violence vs gentleness
    ("Describe emergency surgery with blood and incisions.", "Describe a gentle massage therapy session."),
    ("Write about a dentist pulling a tooth forcefully.", "Write about a dentist carefully cleaning teeth."),
    ("Describe setting a broken bone with audible cracking.", "Describe a physical therapist guiding gentle stretches."),
    ("Write about cauterizing a wound.", "Write about applying a soothing bandage."),
    ("Describe an aggressive chemotherapy treatment.", "Describe a gentle acupuncture session."),
    ("Write about a bone marrow extraction.", "Write about a blood donation."),
    ("Describe stitches being sewn into skin.", "Describe moisturizer being applied gently."),
    ("Write about a forceps delivery.", "Write about a water birth."),
    ("Describe cardioversion shocking a heart back to rhythm.", "Describe meditation slowing a heartbeat to calm."),
    ("Write about debriding burned tissue.", "Write about aloe vera soothing a sunburn."),
    # More conflict
    ("Describe a police officer using a taser on a suspect.", "Describe a police officer de-escalating a tense situation."),
    ("Write about tear gas being deployed at a protest.", "Write about a peaceful protest with singing and chanting."),
    ("Describe rubber bullets being fired at demonstrators.", "Describe demonstrators linking arms in peaceful solidarity."),
    ("Write about a SWAT team breaching a door.", "Write about a social worker talking someone down."),
    ("Describe a character whipping a horse to make it run faster.", "Describe a character gently guiding a horse with reins."),
    ("Write about someone keying a car in revenge.", "Write about someone leaving a polite note on a badly parked car."),
    ("Describe a character smashing a guitar on stage.", "Describe a character playing a gentle acoustic melody."),
    ("Write about someone burning books in a bonfire.", "Write about someone donating books to a library."),
    ("Describe a logger clear-cutting an ancient forest.", "Describe a forester selectively harvesting sustainably."),
    ("Write about overfishing depleting the ocean violently.", "Write about marine conservation restoring fish populations."),
]

with open("dir-steering/examples/violent_v1.txt", "w") as fg, open("dir-steering/examples/peaceful_v1.txt", "w") as fb:
    fg.write(f"# {len(pairs)} violent/aggressive prompts\n")
    fb.write(f"# {len(pairs)} peaceful/calm prompts\n")
    for g, b in pairs[:100]:
        fg.write(g + "\n")
        fb.write(b + "\n")

print(f"Wrote {min(len(pairs), 100)} pairs")
