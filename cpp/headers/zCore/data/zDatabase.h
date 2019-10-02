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

#ifndef ZSPACE_DATABASE_H
#define ZSPACE_DATABASE_H

#pragma once

#include <vector>
#include <string>
#include <algorithm>    // std::sort
using namespace std;

#include<depends/SQLITE/sqlite3.h>

#include <headers/zCore/base/zEnumerators.h>
#include <headers/zCore/base/zInline.h>
#include <headers/zCore/base/zTypeDef.h>

namespace  zSpace
{

	/** \addtogroup zCore
	*	\brief The core datastructures of the library.
	*  @{
	*/

	/** \addtogroup zData
	*	\brief The data classes and structs of the library.
	*  @{
	*/		

	/*! \class zDatabase
	*	\brief A database class for accessing a SQL database using SQLite library.
	*	\since version 0.0.1
	*/

	/** @}*/

	/** @}*/
	
	class ZSPACE_CORE zDatabase
	{
	private:
		//--------------------------
		//---- PRIVATE ATTRIBUTES
		//--------------------------

		/*!	\brief SQL database  */
		sqlite3 * database;

	public:

		//--------------------------
		//---- CONSTRUCTOR
		//--------------------------

		/*! \brief Default constructor.
		*
		*	\since version 0.0.1
		*/
		zDatabase();

		/*! \brief Overloaded constructor.
		*
		*	\param		[in]	filename		- file path to the SQL database.
		*	\since version 0.0.1
		*/		
		zDatabase(char* filename);

		//--------------------------
		//---- DESTRUCTOR
		//--------------------------

		/*! \brief Default destructor.
		*
		*	\since version 0.0.1
		*/	
		~zDatabase();

		//--------------------------
		//---- DATABASE METHODS
		//--------------------------

		/*! \brief This method opens the database at the input file path.
		*
		*	\param		[in]	filename		- file path to the SQL database.
		*	\since version 0.0.1
		*/
		bool open(char* filename);

		/*! \brief This method closes the current database.
		*		
		*	\since version 0.0.1
		*/
		void close();

		/*! \brief This method queries the database with the command given by the input SQL statement.
		*
		*	\param		[in]	sqlStatment		- SQL statements as string container.
		*	\param		[in]	sqlCommandType	- SQL command type.  Refer zSQLCommand for types.
		*	\param		[in]	displayError	- error is displayed in the console if true.
		*	\param		[out]	outStatment		- container of selection if zSQLCommand is zSelect / zSelectCount .
		*	\param		[in]	colType			- container of selection to contain column type if true.
		*	\since version 0.0.1
		*/
		bool sqlCommand(zStringArray &sqlStatment, zSQLCommand sqlCommandType, bool displayError, zStringArray &outStatment, bool colType = true);

		
		/*! \brief This method creates a new table in the database.
		*
		*	\param		[in]	sqlStatment			- SQL statements as string container.
		*	\param		[in]	tableName			- name of new table.
		*	\param		[in]	columnNames			- column names of the table.
		*	\param		[in]	columnTypes			- column types of the table.
		*	\param		[in]	checkPrimaryKey		- true if a primary key is to be created.
		*	\param		[in]	primarykey			- to contain the primary key if checkPrimaryKey is true.
		*	\since version 0.0.1
		*/
		void tableCreate(zStringArray &sqlStatment, string &tableName, zStringArray &columnNames, zStringArray &columnTypes, bool checkPrimaryKey, zStringArray &primarykey);

		/*! \brief This method inserts values into existing table in the database.
		*
		*	\param		[in]	sqlStatment			- SQL statements as string container.
		*	\param		[in]	tableName			- name of new table.
		*	\param		[in]	columnNames			- column names of the table.
		*	\param		[in]	values				- container of valuse to be inserted.
		*	\since version 0.0.1
		*/		
		void tableInsert(zStringArray &sqlStatment, string &tableName, zStringArray& columnNames, zStringArray &values);
		
	
	};
}

#if defined(ZSPACE_STATIC_LIBRARY)  || defined(ZSPACE_DYNAMIC_LIBRARY)
// All defined OK so do nothing
#else
#include<source/zCore/data/zDatabase.cpp>
#endif

#endif