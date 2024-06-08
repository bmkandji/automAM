# Utiliser l'image de base de Python 3.12.0
FROM python:3.12.0

# Installer R et des dépendances pour rpy2
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository 'deb http://cloud.r-project.org/bin/linux/debian buster-cran40/' && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 'E298A3A825C0D65DFD57CBB651716619E084DAB9' && \
    apt-get update && apt-get install -y \
    r-base=4.3.1-1~bpo10+1 \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libv8-dev

# Installer des packages R nécessaires pour rpy2
RUN R -e "install.packages(c('ggplot2', 'dplyr', 'tidyverse', 'data.table', 'shiny', 'reticulate'), repos='https://cran.rstudio.com/')"

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt /workspace/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Définir le répertoire de travail
WORKDIR /workspace

# Copier les fichiers du projet dans le conteneur
COPY . /workspace

# Exposer le port Jupyter Notebook (optionnel)
EXPOSE 8888

# Commande par défaut pour exécuter l'application Python
CMD ["python", "main.py"]
