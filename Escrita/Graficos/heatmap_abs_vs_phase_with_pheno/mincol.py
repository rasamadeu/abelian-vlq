#!/usr/bin/python3
import sys


def main():

    args = sys.argv[1:]

    if not args[1].isnumeric:
        print("Second argument is position of column\n")
        return

    if args[2] == "-h":
        if not args[3].isnumeric:
            print("-h wrong input: number of header lines to ignore\n")
            return
        header_lines = int(args[3])

    with open(args[0], "r") as f:
        for i in range(header_lines + 1):
            line = f.readline()

        min = float(line.split()[int(args[1])])
        min_line = line

        line = f.readline()
        while line != "":
            field = float(line.split()[int(args[1])])
            if field < min:
                min = field
                min_line = line
            line = f.readline()

    print(min_line)
    return


if __name__ == "__main__":
    main()
