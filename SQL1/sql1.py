# sql1.py
"""Volume 3: SQL 1 (Introduction).
<Mingyan Zhao>
<Math 321>
<10/14/2018>
"""

import sqlite3 as sql
import csv
import numpy as np
from matplotlib import pyplot as plt


# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """

    try:
        # Establish a connection to a database file or create one if it doesn't exist.
        with sql.connect(db_file) as conn:
            cur = conn.cursor() # Get a cursor object.
            #Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from the database if they exist.
            cur.execute("DROP TABLE IF EXISTS MajorInfo")
            cur.execute("DROP TABLE IF EXISTS CourseInfo")
            cur.execute("DROP TABLE IF EXISTS StudentInfo")
            cur.execute("DROP TABLE IF EXISTS StudentGrades")
            #add the following tables to the database with the specified column names and types.
            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT)")
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT)")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTERGER)")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT)")

    finally:
        #close the database
        conn.close()

    try:
        #save csv files
        with open(student_info, "r") as info:
            rows_info = list(csv.reader(info))
        with open(student_grades, "r") as grades:
            rows_grades = list(csv.reader(grades))

        MajorID = [(1,'Math'),(2,'Science'),(3,'Writing'),(4,'Art')]
        CourseID = [(1,'Calculus'),(2,'English'),(3,'Pottery'),(4,'History')]

        #updatee the database from the file and add the major ID and Course ID
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", rows_info)
            cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", rows_grades)
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", MajorID)
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", CourseID)

    finally:
        conn.close()
    try:
        #
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #Modify the StudentInfo table, values of −1 in the MajorID column are replaced with NULL values.
            cur.execute("UPDATE StudentInfo SET MajorID=NULL WHERE MajorID=-1;")
    finally:
        conn.close()



# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    try:
        # Establish a connection to a database file or create one if it doesn't exist.
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            cur.execute("DROP TABLE IF EXISTS USEarthquakes")
            cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL)")
    finally:
        #close file
        conn.close()

    try:
        #read the file
        with open(data_file, "r") as info:
            rows_info = list(csv.reader(info))

        with sql.connect(db_file) as conn:
            #update the database
            cur = conn.cursor()
            cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);", rows_info)
    finally:
        conn.close()

    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #1. Remove rows from USEarthquakes that have a value of 0 for the Magnitude.
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude=0;")
            #2. Replace 0 values in the Day, Hour, Minute, and Second columns with NULL values.
            cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day=0;")
            cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour=0;")
            cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute=0;")
            cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second=0;")
    finally:
        conn.close()

# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #query the database for all tuples of the form (StudentName, CourseName) where that student has an “A” or “A+” grade in that course.
            cur.execute("SELECT SI.StudentName, CI.CourseName "
                "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades AS SG "
                "WHERE SI.StudentID == SG.StudentID AND CI.CourseID == SG.CourseID AND (SG.Grade == 'A+' OR SG.Grade == 'A')")

            #Return the list of tuples.
            names = cur.fetchall()
    finally:
        conn.close()
    return names



# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            #The magnitudes of the earthquakes during the 19th century (1800–1899).
            cur.execute("SELECT Magnitude FROM USEarthquakes " " WHERE Year BETWEEN 1800 AND 1899;")
            first = np.ravel(cur.fetchall())
            #The magnitudes of the earthquakes during the 20th century (1900–1999).
            cur.execute("SELECT Magnitude FROM USEarthquakes " " WHERE Year BETWEEN 1900 AND 1999;")
            second = np.ravel(cur.fetchall())
            # The average magnitude of all earthquakes in the database
            cur.execute("SELECT AVG(Magnitude) FROM USEarthquakes;")
            average = np.ravel(cur.fetchall())

    finally:
        conn.close()

    #plotting
    plt.subplot(121)
    plt.hist(first)
    plt.ylabel("years")
    plt.xlabel("magnitudes")
    plt.title("earthquakes during the 19th century")

    plt.subplot(122)
    plt.hist(second)
    plt.ylabel("years")
    plt.xlabel("magnitudes")
    plt.title("earthquakes during the 20th century")
    plt.show()

    return average[0]
