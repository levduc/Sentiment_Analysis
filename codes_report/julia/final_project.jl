
#Multinomial Naive Bayes
##########################################################################
##Learning Process
##Preprocess the Data
##Generate negative review
##open mega negative doc
f = open("NegVocab.txt")
lines = readlines(f)
close(f)
#chomp each line
for i = 1:length(lines)
    lines[i] = chomp(lines[i])
end
#create negative dictionary with tf
NegDictionary = Dict()
for i = 1:length(lines)
    tempArray = split(lines[i])
    tempWord = ""
    tempWord = tempArray[1]
    #tf
    tempTF = parse(tempArray[2])
    NegDictionary[tempWord] = tempTF
end
##build postive dictionary
##open mega positive doc
f = open("PosVocab.txt")
lines = readlines(f)
close(f)
#chomp each line
for i = 1:length(lines)
    lines[i] = chomp(lines[i])
end
#create Post dictionary with tf
PosDictionary = Dict()
for i = 1:length(lines)
    tempArray = split(lines[i])
    tempWord = ""
    tempWord = tempArray[1]
    #tf
    tempTF = parse(tempArray[2])
    PosDictionary[tempWord] = tempTF
end
##build dictionary
##open vocab files
f = open("vocab.txt")
lines = readlines(f)
close(f)
lines
#chomp each line
for i = 1:length(lines)
    lines[i] = chomp(lines[i])
end
#create dictionary with tf
Vocab = Dict()
for i = 1:length(lines)
    tempArray = split(lines[i])
    tempWord = ""
    tempWord = tempArray[1]
    tempTF = parse(tempArray[2])
    Vocab[tempWord] = tempTF
end
###################################################################################################
#read in review
function generateReview(fileName)
    f = open(fileName)
    lines = readlines(f)
    close(f)
    #chomp each line
    review = ""
    for i = 1:length(lines)
        tempString = lines[i]
        tempString = chomp(tempString)
        tempString = replace(tempString,"\"","")
        tempString = replace(tempString,"(","")
        tempString = replace(tempString,")","")
        tempString = replace(tempString,".","")
        tempString = replace(tempString,",","")
        tempString = replace(tempString,":","")
        tempString = replace(tempString,";","")
        tempString = replace(tempString,"  "," ")
        review = review * tempString
    end
    return review
end
###############################################################################################
#dictionary of postive words and negative words
f = open("posnegwords.txt")
lines = readlines(f)
close(f)
PosNegWords = Dict()
for i = 1:length(lines)
    lines[i] = chomp(lines[i])
    lines[i] = replace(lines[i],"\t","  ")
    lines[i] = replace(lines[i],"  "," ")
    tempArray = split(lines[i])
    for w in tempArray
        if(w == "Negativ")
            tempArray[1] = replace(tempArray[1],"#","")
            tempArray[1] = replace(tempArray[1],"1","")
            tempArray[1] = replace(tempArray[1],"2","")
            tempArray[1] = replace(tempArray[1],"3","")
            tempArray[1] = replace(tempArray[1],"4","")
            tempArray[1] = replace(tempArray[1],"5","")
            tempArray[1] = replace(tempArray[1],"6","")
            tempArray[1] = replace(tempArray[1],"7","")
            tempArray[1] = replace(tempArray[1],"8","")
            tempArray[1] = replace(tempArray[1],"9","")
            tempArray[1] = replace(tempArray[1],"10","")
            PosNegWords[lowercase(tempArray[1])] = -1
        end
        if(w == "Positiv")
            tempArray[1] = replace(tempArray[1],"#","")
            tempArray[1] = replace(tempArray[1],"1","")
            tempArray[1] = replace(tempArray[1],"2","")
            tempArray[1] = replace(tempArray[1],"3","")
            tempArray[1] = replace(tempArray[1],"4","")
            tempArray[1] = replace(tempArray[1],"5","")
            tempArray[1] = replace(tempArray[1],"6","")
            tempArray[1] = replace(tempArray[1],"7","")
            tempArray[1] = replace(tempArray[1],"8","")
            tempArray[1] = replace(tempArray[1],"9","")
            tempArray[1] = replace(tempArray[1],"10","")
            PosNegWords[lowercase(tempArray[1])] = 1
        end
    end
end
###################################################################################################
#Learning
#compute parameter
function ComputeParaEst(NegDictionary,PosDictionary,Vocab)
    VocabSize = length(Vocab)
    termGNeg = Dict()
    termGPos = Dict()
    NegSize = 0 
    PosSize = 0 
    for (k,tf) in NegDictionary
        NegSize = NegSize + tf
    end
    
    for (k,tf) in PosDictionary
        PosSize = PosSize + tf
    end
    
    for (k,tf) in Vocab
        nk1 = try NegDictionary[k] catch 0 end #find neg words
        nk2 = try PosDictionary[k] catch 0 end #find pos words
#         NP = try PosNegWords[k] catch 0 end
#         if (NP != 0)
#             if (NP == 1)  nk2 = nk2*100 end #upshoot positve words
#             if (NP == -1) nk1 = nk1*100 end #upshoot negative words
#         end
        pwc1 =(nk1+1)/(NegSize+VocabSize) #prob word given class 1 
        pwc2 =(nk2+1)/(PosSize+VocabSize) #prob word given class 2
        termGNeg[k] = pwc1
        termGPos[k] = pwc2
    end
    return termGNeg,termGPos
end
termGNeg,termGPos = ComputeParaEst(NegDictionary,PosDictionary,Vocab);

#MULTINOMIAL NAIVE BAYES
function NaiveBayes(review,termGNeg,termGPos,NegDictionary,
                    PosDictionary,Vocab,PosNegWords)
    #VocabSize
    VocabSize = length(termGNeg)
    #break down review into word.
    listOfWord = split(review)
    #prior 
    pN = 1/2
    pP = 1/2
    scoreN = log10(pN)
    scoreP = log10(pP)
    #spagetti code, compute negvocab size
    NegSize = 0 
    PosSize = 0 
    for (k,tf) in NegDictionary
        NegSize = NegSize + tf
    end
    
    for (k,tf) in PosDictionary
        PosSize = PosSize + tf
    end
    #compute
    for w in listOfWord
        wN = try termGNeg[w] catch 1/(NegSize+VocabSize) end #dealing with unknown words
        wP = try termGPos[w] catch 1/(PosSize+VocabSize) end#dealing with unknown words
        #upweight
        NP = try PosNegWords[w] catch 0 end
        if (NP != 0)
            if (NP == 1)  wP = wP*2 end #upshoot positve words
            if (NP == -1) wN = wN*2 end #upshoot negative words
        end
        #end upweight
        scoreN = scoreN + log10(wN) 
        scoreP = scoreP + log10(wP) 
    end
    if (scoreN > scoreP) 
        return "-"
    else    
        return "+"
    end
end

NegCount = 0
PosCount = 0
half = 6000 ###choosing how many reviews to evaluate
size = half*2
for i = 1:half
    fileName ="test_set1\\pos\\pos ($i).txt"
    try
        review = generateReview(fileName)
        r = NaiveBayes(review,termGNeg,termGPos,NegDictionary,PosDictionary,Vocab,PosNegWords)
        if (r == "+")
            PosCount +=1
        else
            NegCount +=1
        end
    catch
        println(i)
    end
end
NegCount1 = 0
PosCount1 = 0
for i = 1:half
    fileName ="test_set1\\neg\\neg ($i).txt"
    try
        review = generateReview(fileName)
        r = NaiveBayes(review,termGNeg,termGPos,NegDictionary,PosDictionary,Vocab,PosNegWords)
        if (r == "+")
            PosCount1 +=1
        else
            NegCount1 +=1
        end
    catch
        println(i)
    end
end

(PosCount + NegCount1)/size #compute accuracy

println(PosCount,"\t",size-PosCount)

#sample outpur
for i = 1:20
    fileName ="test_set1\\pos\\pos ($i).txt"
    review = generateReview(fileName)
    println(i,"\t\t",NaiveBayes(review,termGNeg,termGPos,NegDictionary,PosDictionary,Vocab,PosNegWords))
end

################################################################################
#Binarized Naive Bayes
##Preprocess the Data
##Generate negative review
##open mega negative doc
f = open("NegVocab.txt")
lines = readlines(f)
close(f)
#chomp each line
for i = 1:length(lines)
    lines[i] = chomp(lines[i])
end
#create negative dictionary with tf
NegDictionary = Dict()
for i = 1:length(lines)
    tempArray = split(lines[i])
    tempWord = ""
    tempWord = tempArray[1]
    #df
    tempDF = parse(tempArray[3])
    NegDictionary[tempWord] = tempDF
end
##build postive dictionary
##open mega positive doc
f = open("PosVocab.txt")
lines = readlines(f)
close(f)
#chomp each line
for i = 1:length(lines)
    lines[i] = chomp(lines[i])
end
#create Post dictionary with tf
PosDictionary = Dict()
for i = 1:length(lines)
    tempArray = split(lines[i])
    tempWord = ""
    tempWord = tempArray[1]
    #df
    tempDF = parse(tempArray[3])
    PosDictionary[tempWord] = tempDF
end
##build dictionary
##open vocab files
f = open("vocab.txt")
lines = readlines(f)
close(f)
lines
#chomp each line
for i = 1:length(lines)
    lines[i] = chomp(lines[i])
end
#create dictionary with tf
Vocab = Dict()
for i = 1:length(lines)
    tempArray = split(lines[i])
    tempWord = ""
    tempWord = tempArray[1]
    #df
    tempDF = parse(tempArray[3])
    Vocab[tempWord] = tempDF
end
###################################################################################################
#Learning
#compute parameter
function ComputeParaEst(NegDictionary,PosDictionary,Vocab)
    VocabSize = length(Vocab)
    termGNeg = Dict()
    termGPos = Dict()
    NegSize = 0 
    PosSize = 0 
    for (k,tf) in NegDictionary
        NegSize = NegSize + tf
    end
    
    for (k,tf) in PosDictionary
        PosSize = PosSize + tf
    end
    
    for (k,tf) in Vocab
        nk1 = try NegDictionary[k] catch 0 end #find neg words
        nk2 = try PosDictionary[k] catch 0 end #find pos words
        pwc1 =(nk1+1)/(NegSize+VocabSize) #prob word given class 1 
        pwc2 =(nk2+1)/(PosSize+VocabSize) #prob word given class 2
        termGNeg[k] = pwc1
        termGPos[k] = pwc2
    end
    return termGNeg,termGPos
end
termGNeg,termGPos = ComputeParaEst(NegDictionary,PosDictionary,Vocab);

#Binary multinomial Naive Bayes
function BinarizedNaiveBayes(review,termGNeg,termGPos,NegDictionary,
                             PosDictionary,Vocab,PosNegWords)
    #VocabSize
    VocabSize = length(termGNeg)
    #break down review into word.
    listOfWord = split(review)
    reviewDict = Dict()
    #for each word review
    for w in  listOfWord
        reviewDict[w] = 1
    end
    #prior 
    pN = 1/2
    pP = 1/2
    scoreN = log10(pN)
    scoreP = log10(pP)
    #spagetti code, compute negvocab size
    NegSize = 0 
    PosSize = 0 
    for (k,tf) in NegDictionary
        NegSize = NegSize + tf
    end
    
    for (k,tf) in PosDictionary
        PosSize = PosSize + tf
    end
    
    #compute
    for (w,v) in reviewDict
        wN = try termGNeg[w] catch 1/(NegSize+VocabSize) end #dealing with unknown words
        wP = try termGPos[w] catch 1/(PosSize+VocabSize) end #dealing with unknown words
        #upweight
#         NP = try PosNegWords[w] catch 0 end
#         if (NP != 0)
#             if (NP == 1)  wP = wP*3 end #upshoot positve words
#             if (NP == -1) wN = wN*3 end #upshoot negative words
#         end
        #upweight
        scoreN = scoreN + log10(wN) 
        scoreP = scoreP + log10(wP) 
    end
    if (scoreN > scoreP) 
        return "-"
    else    
        return "+"
    end
end

NegCount = 0
PosCount = 0
half = 3200
size = half*2
for i = 1:half
    fileName ="test_set1\\pos\\pos ($i).txt"
    try
        review = generateReview(fileName)
        r = BinarizedNaiveBayes(review,termGNeg,termGPos,NegDictionary,PosDictionary,Vocab,PosNegWords)
        if (r == "+")
            PosCount +=1
        else
            NegCount +=1
        end
    catch
        println(i)
    end
end

NegCount1 = 0
PosCount1 = 0
for i = 1:half
    fileName ="test_set1\\neg\\neg ($i).txt"
    try
        review = generateReview(fileName)
        r = BinarizedNaiveBayes(review,termGNeg,termGPos,NegDictionary,PosDictionary,Vocab,PosNegWords)
        if (r == "+")
            PosCount1 +=1
        else
            NegCount1 +=1
        end
    catch
        println(i)
    end
end

(PosCount + NegCount1)/size

println(PosCount,"\t",size-PosCount)

x = [200;800;1600;3200;6400;12000];
y1 = [.695;.676;.642;.622;.614;616];
y2 = [.765;.771;.733;.714;.703;.711];
y3 = [.790;.760;.710;.700;.708;.709];
y4 = [.840;.830;.799;.789;.794;.793];

using Plots
plot(x,y1,marker=([:hex :d]),lab="NB")
plot!(x,y2,marker=([:hex :d]),lab="BNB")
plot!(x,y3,marker=([:hex :d]),lab="UNB")
plot!(x,y4,marker=([:hex :d]),lab="CNB")
savefig("Accuracy.pdf")
