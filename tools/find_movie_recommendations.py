"""
This is an example tool to demonstrate how to create custom tools for our agent.

This tool is used to find movie recommendations based on a movie title and rating criteria.
"""

import os
from typing import Any

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase, RoutingControl
from pydantic import BaseModel, Field


def find_movie_recommendations(
    movie_title: str, min_user_rating: float = 4.0, limit: int = 10
) -> list[dict[str, Any]]:
    """
    Search the database movie recommendations based on movie title and rating criteria.
    """

    query = """
MATCH (target:Movie)
WHERE target.title = $movieTitle
MATCH (target)<-[r1:RATED]-(u:User)
WHERE r1.rating >= $minRating
MATCH (u)-[r2:RATED]->(similar:Movie)
WHERE similar <> target 
  AND r2.rating >= $minRating 
  AND similar.imdbRating IS NOT NULL
WITH similar, count(*) as supporters, avg(r2.rating) as avgRating
WHERE supporters >= 10
RETURN similar.title, similar.year, similar.imdbRating, 
       supporters as people_who_loved_both, 
       round(avgRating, 2) as avg_rating_by_target_lovers
ORDER BY supporters DESC, avgRating DESC
LIMIT $limit
    """

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    results = driver.execute_query(
        query,
        parameters_={"movieTitle": movie_title, "minRating": min_user_rating, "limit": limit},
        database_=os.getenv("NEO4J_DATABASE"),
        routing_=RoutingControl.READ,
        result_transformer_=lambda r: r.data(),
    )
    return results


class FindMovieRecommendationsInput(BaseModel):
    movie_title: str = Field(
        ...,
        description="The title of the movie to find recommendations for. If beginning with 'The', then will follow format of 'Title, The'.",
    )
    min_user_rating: float = Field(
        default=4.0,
        description="The minimum rating of the movie to find recommendations for. ",
        ge=0.5,
        le=5.0,
    )
    limit: int = Field(
        default=10,
        description="The maximum number of recommendations to return. ",
        ge=1,
    )


find_movie_recommendations_tool = StructuredTool.from_function(
    func=find_movie_recommendations,  #           -> The function that the tool calls when executed
    # name=...,                                   -> this is populated by the function name
    # description=...,                            -> this is populated by the function docstring
    args_schema=FindMovieRecommendationsInput,  # -> The input schema for the tool
    return_direct=False,  #                       -> Whether to return the raw result to the user
    # coroutine=...,                              -> An async version of the function
)
