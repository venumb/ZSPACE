// This file is part of zspace, a simple C++ collection of geometry data-structures & algorithms, 
// data analysis & visualization framework.
//
// Copyright (C) 2019 ZSPACE 
// 
// This Source Code Form is subject to the terms of the MIT License 
// If a copy of the MIT License was not distributed with this file, You can 
// obtain one at https://opensource.org/licenses/MIT.
//
// Author : Vishu Bhooshan <vishu.bhooshan@zaha-hadid.com>
//

#include<headers/zCore/data/zDatabase.h>

namespace zSpace
{

	//---- CONSTRUCTOR

	ZSPACE_INLINE zDatabase::zDatabase()
	{
		database = NULL;
	}

	ZSPACE_INLINE zDatabase::zDatabase(char* filename)
	{
		database = NULL;
		bool stat = open(filename);

		if (stat) printf("\n Opened database successfully\n");
		else printf("\n Can't open database \n");

	}

	//---- DESTRUCTOR

	ZSPACE_INLINE zDatabase::~zDatabase() {}

	//---- DATABASE METHODS

	ZSPACE_INLINE bool zDatabase::open(char* filename)
	{
		if (sqlite3_open(filename, &database) == SQLITE_OK)
			return true;

		return false;
	}

	ZSPACE_INLINE void zDatabase::close()
	{
		sqlite3_close(database);
	}

	ZSPACE_INLINE bool zDatabase::sqlCommand(zStringArray &sqlStatment, zSQLCommand sqlCommandType, bool displayError, zStringArray &outStatment, bool colType)
	{
		outStatment.clear();

		bool out = false;
		int RC;
		int extended_RC;

		char* sql;

		sqlite3_stmt * Selectionstmt;
		int arraySize = 0;

		// get array size
		for (int i = 0; i < sqlStatment.size(); i++)
		{
			int lenS = strlen(sqlStatment[i].c_str());
			arraySize += lenS;
		}


		// intialise dyanmic array
		sql = new char[arraySize];
		int counter = 0;

		//printf("\n ");
		for (int i = 0; i < sqlStatment.size(); i++)
		{

			int lenS = strlen(sqlStatment[i].c_str());


			for (int j = 0; j < lenS; j++)
			{
				sql[counter] = sqlStatment[i].c_str()[j];

				//printf("%c", sqlStatment[i].c_str()[j]);
				counter++;
			}
			//printf("\n ");
		}




		/* Execute SQL statement */
		if (sqlCommandType != zSelect && sqlCommandType != zSelectCount) RC = sqlite3_exec(database, sql, NULL, NULL, NULL);

		else
		{
			sqlite3_stmt * stmt;
			const char *pzTail;
			//sqlite3_exec(database, "BEGIN TRANSACTION", 0, 0, 0);

			sqlite3_prepare_v2(database, sql, -1, &stmt, &pzTail);//preparing the statement
																  //printf(" %s", &pzTail);

			RC = sqlite3_step(stmt);
			sqlite3_finalize(stmt);
		}


		if (sqlCommandType != zSelect && sqlCommandType != zSelectCount)
		{
			if (RC == SQLITE_DONE || RC == SQLITE_OK)out = true;
			else 	extended_RC = sqlite3_extended_errcode(database);
		}
		else
		{
			if (RC == SQLITE_ROW) out = true;
			else 	extended_RC = sqlite3_extended_errcode(database);
		}

		// Display information

		if (displayError)
		{
			if (out)
			{
				if (sqlCommandType == zCreate) printf("\n Table Creation Succesful,");
				if (sqlCommandType == zInsert) printf("\n Table Insert Succesful,");
				if (sqlCommandType == zSelect) printf("\n Table Selection Succesful,");
				if (sqlCommandType == zSelectExists)	printf("\n Table Selection Exists,");
				if (sqlCommandType == zSelectCreate)	printf("\n Table Selection & Creation Succesful,");
				if (sqlCommandType == zUpdate)	printf("\n Table Update Succesful,");
				if (sqlCommandType == zDrop)	printf("\n Table Drop Succesful,");
				if (sqlCommandType == zExportCSV)	printf("\n CSV Export Succesful,");
				if (sqlCommandType == zSelectCount) printf("\n Table Row Counts Succesful,");

				printf(" RC: %i \n", RC);

			}
			else
			{
				if (sqlCommandType == zCreate) printf("\n Table Creation Failed,");
				if (sqlCommandType == zInsert) printf("\n Table Insert Failed,");
				if (sqlCommandType == zSelect) printf("\n Table Selection Failed,");
				if (sqlCommandType == zSelectExists)	printf("\n Table Selection Doesnt Exists,");
				if (sqlCommandType == zSelectCreate)	printf("\n Table Selection & Creation Failed,");
				if (sqlCommandType == zUpdate)	printf("\n Table Update Failed,");
				if (sqlCommandType == zDrop)	printf("\n Table Drop Failed,");
				if (sqlCommandType == zExportCSV)	printf("\n CSV Export Failed,");
				if (sqlCommandType == zSelectCount) printf("\n Table Row Counts Failed,");

				printf(" RC: %i,", RC);


				if (extended_RC == 1555) printf(" Primary Key Constraint Failed,", extended_RC);

				printf(" ext_RC: %i \n", extended_RC);
			}


			// print selection data

			if (sqlCommandType == zSelect || sqlCommandType == zSelectCount)
			{
				if (out)
				{
					sqlite3_prepare_v2(database, sql, -1, &Selectionstmt, NULL);//preparing the statement
					int row = 0;
					while (1)
					{
						int s;
						s = sqlite3_step(Selectionstmt);
						if (s == SQLITE_ROW) {
							int bytes;
							const unsigned char * text;
							const char * columnType;

							int cols = sqlite3_data_count(Selectionstmt);
							printf("%d:", row);

							//outStatment.push_back(to_string(row)); // row Number
							for (int i = 0; i < cols; i++)
							{
								bytes = sqlite3_column_bytes(Selectionstmt, i);
								text = sqlite3_column_text(Selectionstmt, i);


								outStatment.push_back(reinterpret_cast<const char*>(text)); // data string
								if (sqlCommandType == zSelect && colType)
								{
									columnType = sqlite3_column_decltype(Selectionstmt, i);
									outStatment.push_back(columnType); // column Type
									printf(":%s (%s)", text, columnType);
								}


							}
							printf("\n");

							row++;
						}
						else if (s == SQLITE_DONE) {
							break;
						}
						else {
							fprintf(stderr, "Failed.\n");
							exit(1);
						}

					}

					sqlite3_finalize(Selectionstmt);
				}

			}
		}
		else
		{
			if (sqlCommandType == zSelect || sqlCommandType == zSelectCount)
			{
				if (out)
				{
					sqlite3_prepare_v2(database, sql, -1, &Selectionstmt, NULL);//preparing the statement
					int row = 0;
					while (1)
					{
						int s;
						s = sqlite3_step(Selectionstmt);
						if (s == SQLITE_ROW) {
							int bytes;
							const unsigned char * text;
							const char * columnType;

							int cols = sqlite3_data_count(Selectionstmt);
							//printf("%d:", row);

							//outStatment.push_back(to_string(row)); // row Number

							for (int i = 0; i < cols; i++)
							{
								bytes = sqlite3_column_bytes(Selectionstmt, i);
								text = sqlite3_column_text(Selectionstmt, i);
								columnType = sqlite3_column_decltype(Selectionstmt, i);

								if (sqlCommandType == zSelect && colType)
								{
									columnType = sqlite3_column_decltype(Selectionstmt, i);
									outStatment.push_back(columnType); // column Type
								}

								outStatment.push_back(reinterpret_cast<const char*>(text)); // data string

																							//printf(":%s (%s)", text, columnType);
							}
							//printf("\n");

							row++;
						}
						else if (s == SQLITE_DONE) {
							break;
						}
						else {
							fprintf(stderr, "Failed.\n");
							exit(1);
						}

					}

					sqlite3_finalize(Selectionstmt);
				}
			}
		}

		return out;
	}

	ZSPACE_INLINE void zDatabase::tableCreate(zStringArray &sqlStatment, string &tableName, zStringArray &columnNames, zStringArray &columnTypes, bool checkPrimaryKey, zStringArray &primarykey)
	{
		sqlStatment.push_back("CREATE TABLE ");  // table command
		sqlStatment.push_back("IF NOT EXISTS "); // Check if exists command
		sqlStatment.push_back(tableName); // table Name

		sqlStatment.push_back("(");  // table headers

		for (int i = 0; i < columnNames.size(); i++)
		{
			sqlStatment.push_back(" " + columnNames[i] + " " + columnTypes[i] + " ");
			if (i != columnNames.size() - 1)sqlStatment.push_back(", ");
		}

		if (checkPrimaryKey)
		{
			sqlStatment.push_back(", PRIMARY KEY(");
			for (int i = 0; i < primarykey.size(); i++)
			{
				sqlStatment.push_back(" " + primarykey[i]);
				if (i != primarykey.size() - 1)sqlStatment.push_back(", ");
			}
			sqlStatment.push_back(")");
		}
		sqlStatment.push_back(");");
	}

	ZSPACE_INLINE void zDatabase::tableInsert(zStringArray &sqlStatment, string &tableName, zStringArray& columnNames, zStringArray &values)
	{
		sqlStatment.push_back("INSERT INTO ");  // insert command
		sqlStatment.push_back(tableName);  // table command
		sqlStatment.push_back("( ");

		for (int i = 0; i < columnNames.size(); i++)
		{
			sqlStatment.push_back(" " + columnNames[i] + " ");
			if (i != columnNames.size() - 1)sqlStatment.push_back(", ");


		}

		sqlStatment.push_back(") VALUES (");;

		for (int i = 0; i < values.size(); i++)
		{
			sqlStatment.push_back(" " + values[i] + " ");
			if (i != values.size() - 1)sqlStatment.push_back(", ");
		}

		sqlStatment.push_back(");");
	}
}