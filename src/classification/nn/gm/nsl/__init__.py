
def load_brown_clusters_mapping(brownMappingFile):
    b2cl = {}
    f = open(brownMappingFile)
    for line in f:
        parts = line.split('\t')
        b2cl[parts[1]] = parts[0]
    return b2cl
        
