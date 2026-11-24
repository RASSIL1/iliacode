from datetime import date


def tickets_du_mois(tickets, annee, mois):
    resultat = []
    for ticket in tickets:
        jour = date.fromisoformat(ticket["ouverture"])
        if jour.year == annee and jour.month == mois:
            resultat.append(ticket)
    return resultat


def resume_par_equipe(tickets):
    resume = {}
    for ticket in tickets:
        equipe = ticket.get("equipe", "sans equipe")
        ligne = resume.setdefault(equipe, {"ouverts": 0, "fermes": 0})
        if ticket["statut"] == "ferme":
            ligne["fermes"] += 1
        else:
            ligne["ouverts"] += 1
    return resume


def delai_moyen_de_fermeture(tickets):
    delais = []
    for ticket in tickets:
        if ticket["statut"] != "ferme":
            continue
        ouverture = date.fromisoformat(ticket["ouverture"])
        fermeture = date.fromisoformat(ticket["fermeture"])
        delais.append((fermeture - ouverture).days)
    if not delais:
        return 0
    return sum(delais) / len(delais)


def lignes_du_rapport(tickets, annee, mois):
    lignes = [f"Rapport des tickets, {mois:02d}/{annee}"]
    du_mois = tickets_du_mois(tickets, annee, mois)
    lignes.append(f"Tickets ouverts dans le mois : {len(du_mois)}")
    for equipe, compte in sorted(resume_par_equipe(du_mois).items()):
        lignes.append(f"  {equipe} : {compte['ouverts']} ouverts, {compte['fermes']} fermés")
    delai = delai_moyen_de_fermeture(du_mois)
    lignes.append(f"Délai moyen de fermeture : {delai:.1f} jours")
    return lignes


def ecrire_rapport(tickets, annee, mois, chemin):
    with open(chemin, "w", encoding="utf-8") as fichier:
        for ligne in lignes_du_rapport(tickets, annee, mois):
            fichier.write(ligne + "\n")
