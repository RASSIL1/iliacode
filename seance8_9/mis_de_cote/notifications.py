SEUIL_RETARD = 7


def destinataires(ticket, utilisateurs):
    noms = {ticket["auteur"], ticket.get("assigne")}
    return [u for u in utilisateurs if u["nom"] in noms and u.get("actif", True)]


def message_de_changement(ticket, ancien_statut):
    titre = ticket["titre"]
    nouveau = ticket["statut"]
    if nouveau == "ferme":
        return f"Le ticket « {titre} » a été fermé."
    return f"Le ticket « {titre} » est passé de {ancien_statut} à {nouveau}."


def tickets_a_relancer(tickets, jour):
    a_relancer = []
    for ticket in tickets:
        if ticket["statut"] == "ferme":
            continue
        if jour - ticket["derniere_activite"] > SEUIL_RETARD:
            a_relancer.append(ticket)
    return a_relancer


def preparer_envois(tickets, utilisateurs, jour):
    envois = []
    for ticket in tickets_a_relancer(tickets, jour):
        texte = f"Relance, le ticket « {ticket['titre']} » attend depuis plus de {SEUIL_RETARD} jours."
        for utilisateur in destinataires(ticket, utilisateurs):
            envois.append({"a": utilisateur["email"], "texte": texte})
    return envois


def regrouper_par_destinataire(envois):
    boites = {}
    for envoi in envois:
        boites.setdefault(envoi["a"], []).append(envoi["texte"])
    return boites
