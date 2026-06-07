def translation(phrase):
    translation = ""
    for letter in phrase:
        if letter in "AEIOUaeiou":
            translation += "g"
        else:
            translation += letter
    return translation
print(translation(input("enter the phrase: ")))