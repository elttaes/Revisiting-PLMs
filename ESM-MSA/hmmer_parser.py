#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from re import split
from itertools import groupby
from operator import itemgetter

__author__ = 'Enzo Guerrero-Araya (biologoenzo@gmail.com)'
__version__ = '0.1'
__date__ = 'July 13, 2016'

class HMMparser(object):
    """Parser of --domtblout of HMMER
       Default values of Evalue and coverage was taken from dbCAN.
       http://csbl.bmb.uga.edu/dbCAN/download/readme.txt

       ** About what E-value and Coverage cutoff thresholds you should use, we
       have done some evaluation analyses using arabidopsis, rice, Aspergillus
       nidulans FGSC A4, Saccharomyces cerevisiae S288c and Escherichia
       coli K-12 MG1655, Clostridium thermocellum ATCC 27405 and
       Anaerocellum thermophilum DSM 6725. Our suggestion is that
       for plants, use E-value < 1e-23 and coverage > 0.2;
       for bacteria, use E-value < 1e-18 and coverage > 0.35;
       and for fungi, use E-value < 1e-17 and coverage > 0.45."""

    def __init__(self, HMMfile):
        self.HMMfile = HMMfile
        try:
            self.data_HMMfile = open(self.HMMfile).read()
        except FileNotFoundError as e:
            print("The name file or folder is incorrect.\
                  An error has occurred.")
            raise e
        self.parameters = {}  # dict_keys(["Target file", "Option settings", "Program", "Version", "Date", "Current dir", "Pipeline mode", "Query file"])
        for i, line in enumerate(self.data_HMMfile.split("\n")[-10:-2]):
            line = line.split(":")
            key = line[0][2:]
            value = line[1].strip()
            self.parameters[key] = value
        if self.parameters["Program"] == "hmmscan":
            self.matrix = self.hmmscanParser()
        elif self.parameters["Program"] == "hmmsearch":
            self.matrix = self.hmmsearchParser()

    def filterByEvalue(self, evalue=1e-18):
        if self.parameters["Program"] == "hmmscan":
            for i, row in enumerate(self.matrix):
                if float(row[12]) > evalue:  # domain[11] -> i-Evalue (whole database)
                    self.matrix.pop(i)
        elif self.parameters["Program"] == "hmmsearch":
            for i, row in enumerate(self.matrix):
                if float(row[7]) > evalue:  # domain[11] -> Evalue of domine
                    self.matrix.pop(i)

    def filterByBitscore(self, bits=50):
        if self.parameters["Program"] == "hmmscan":
            for i, row in enumerate(self.matrix):
                if float(row[13]) < bits:  # domain[13] -> Bitscore
                    self.matrix.pop(i)
        elif self.parameters["Program"] == "hmmsearch":
            for i, row in enumerate(self.matrix):
                if float(row[8]) < bits:  # domain[8] -> Bitscore
                    self.matrix.pop(i)

    def filterByCoverage(self, cov=0.35):  # covered fraction of HMM
        if self.parameters["Program"] == "hmmscan":
            for i, row in enumerate(self.matrix):
                coverage = (float(row[16]) - float(row[15])) / float(row[2])
                if coverage < cov:  # domain[13] -> Bitscore
                    self.matrix.pop(i)
        elif self.parameters["Program"] == "hmmsearch":
            print("This type of filter due to technical stuff is only \
                   available for hmmscan program, please rerun your hmmsearch \
                   as hmmscan if you need this filter")

    def uniqueByBestBitscore(self,):  # by domain
        if self.parameters["Program"] == "hmmscan":
            matrix = sorted(self.matrix, key=itemgetter(3), reverse=True)
            for query, group in groupby(matrix, itemgetter(3)):
                group = list(group)
                if len(group) == 1:
                    continue
                group = sorted(group, key=itemgetter(13), reverse=True)
                for dom in group[1:]:
                    index = self.matrix.index(dom)
                    self.matrix.pop(index)
        elif self.parameters["Program"] == "hmmsearch":
            print("This type of filter due to technical stuff is only \
                   available for hmmscan program, please rerun your hmmsearch \
                   as hmmscan if you need this filter")

    def hmmscanParser(self, ):
        matrix = []
        # header = ["target name", "accession", "tlen", "query name",
        #           "accession", "qlen", "E-value", "score", "bias", "#", "of",
        #           "c-Evalue", "i-Evalue", "score", "bias", "from", "to",
        #           "from", "to", "from", "to", "acc", "description of target"]
        # matrix.append(header)
        for line in self.data_HMMfile.split("\n"):
            if line.startswith("#") or line is "":
                continue
            line = split("\s+", line, 22)  # just 22 because the last can contain \s+ characters
            matrix.append(line)
        return matrix

    def hmmsearchParser(self, ):
        matrix = []
        # header = ["target name", "accession", "query name", "accession",
        #           "E-value", "score", "bias", "E-value", "score", "bias",
        #           "exp", "reg", "clu", "ov", "env", "dom", "rep", "inc",
        #           "description of target"]
        # matrix.append(header)
        for line in self.data_HMMfile.split("\n"):
            if line.startswith("#") or line is "":
                continue
            line = split("\s+", line, 18)  # just 18 because the last can contain \s+ characters
            matrix.append(line)
        return matrix

# Module test
if __name__ == "__main__":
    import argparse
    import sys
    usage = """%(prog)s reads .domtblout file and returns a custom filtred \
               result (by Evalue, Bitscore, Coverage of HMM model) or it can \
               show only the Best Bitscore domain for each query on hmmscan \
               program output"""

    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument("-b", "--bits", dest="bits",
                        help="The minimum Bitscore threshold to considerate a \
                        domain as a truly hits [be careful with false \
                        positives] (recomended: 50)", type=float)
    parser.add_argument("-e", "--evalue", dest="evalue",
                        help="The maximun E-value threshold to considerate a \
                        domain as a truly hits [be careful with false \
                        positives] (recomended: 1e-18)", type=float)
    parser.add_argument("-c", "--cov", dest="cov",
                        help="The minimum coverage threshold to considerate a \
                        domain as a truly hits [be careful with false \
                        positives] (recomended: 0.35)", type=float)
    parser.add_argument("-u", "--unique", dest="unique", action="store_true",
                        help="show only the Best Bitscore domain for each \
                        query on hmmscan program output", default=False)
    parser.add_argument("-s", "--spacer", dest="spacer", help="Select the \
                        spacer for the program output", default="\t")
    parser.add_argument("-o", "--outfile", nargs="?", help="(default: stdout)",
                        type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument("-v", "--version", action="version", version="%(prog)s v{} ({}) By: {}".format(__version__, __date__, __author__))

    parser.add_argument("domtblout", type=str)

    args = parser.parse_args()
    bits = args.bits
    evalue = args.evalue
    cov = args.cov
    outfile = args.outfile
    domtblout = args.domtblout
    spacer = args.spacer
    hmm = HMMparser(domtblout)

    if args.bits:
        hmm.filterByBitscore(bits)
    if args.evalue:
        hmm.filterByEvalue(evalue)
    if args.cov:
        hmm.filterByCoverage(cov)
    if args.unique:
        hmm.uniqueByBestBitscore()

    if hmm.parameters["Program"] == "hmmscan":
        header = ["target name", "accession", "tlen", "query name",
                  "accession", "qlen", "E-value", "score", "bias", "#", "of",
                  "c-Evalue", "i-Evalue", "score", "bias", "from", "to",
                  "from", "to", "from", "to", "acc", "description of target"]
    else:
        header = ["target name", "accession", "query name", "accession",
                  "E-value", "score", "bias", "E-value", "score", "bias",
                  "exp", "reg", "clu", "ov", "env", "dom", "rep", "inc",
                  "description of target"]
    print(spacer.join(header), file=outfile)
    for row in hmm.matrix:
        row = spacer.join(row)
        print(row, file=outfile)
