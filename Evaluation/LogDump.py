from Tooling import Tools

class LogDumper(Tools):

    def __init__(self):
        self._S = " | "

    def __MakeTable(self, inpt, titles, MinMaxDict = None):

        def AlignColumns(inpt):
            size = 0 
            for i in inpt:
                tmp = max([len(k) for k in i.split(self._S)])
                size = tmp if tmp > size else size
            output = []
            for i in inpt:
                new_str = i.split(self._S)
                output += [self._S.join([" "*(size - len(s)) + s for s in new_str])]
            return output, size

        segment = {}
        segment_H = {}
        xheading = list(inpt)[0]
        yheading = " | ".join([titles[i] + " (" + inpt[xheading][i] + ")" for i in range(len(inpt[xheading]))])
        output = [xheading + self._S + yheading]
        xData = list(inpt)[1:]
        for i in xData:
            output += [str(i)  + self._S + self._S.join([str(j) for j in inpt[i]])]
        output, size = AlignColumns(output)
        
        if MinMaxDict == None:
            return output
        
        Min = "Min" + "".join([self._S + str(MinMaxDict[x][1]) + " @ " +str(MinMaxDict[x][0]) for x in MinMaxDict])
        Max = "Max" + "".join([self._S + str(MinMaxDict[x][3]) + " @ " +str(MinMaxDict[x][2]) for x in MinMaxDict])
        output.insert(1, Min)
        output.insert(2, Max)
        output, size = AlignColumns(output)
        output.insert(1, "_"*len(output[0]))
        output.insert(4, "-"*len(output[0])) 

        return output

    def DumpTLine(self, fig):
        x = fig.xData
        y = fig.yData
        if fig.DoStatistics:
            x = list(x)
        dic = {}
        dic[fig.xTitle] = fig.yTitle
        for i, j in zip(x, y):
            dic[i] = j

        MinMaxDict = {}
        Min = min(y)
        Max = max(y)
        IndexMin = x[y.index(Min)]
        IndexMax = x[y.index(Max)]
        MinMaxDict[fig.Title] = [IndexMin, Min, IndexMax, Max]
        return dic, MinMaxDict

    def MergeDicts(self, dics, MinMaxDict = None):
        titles = list(dics) 
        Cols = {}
        for t in titles:
            keys = list(dics[t])
            for k in keys:
                if k not in Cols:
                    Cols[k] = []
                Cols[k] += [dics[t][k]]
        return self.__MakeTable(Cols, titles, MinMaxDict)

    def DumpTLines(self, figs):
        lines = {}
        MinMax = {}
        for i in range(len(figs)):
            lines[figs[i].Title], MinMaxDict = self.DumpTLine(figs[i])
            MinMax |= MinMaxDict
        return self.MergeDicts(lines, MinMax)

