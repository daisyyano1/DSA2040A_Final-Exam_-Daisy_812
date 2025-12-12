
# OLAP Query: Total Sales by Country and Quarter
SELECT 
    c.ship_country AS Country,
    t.year AS Year,
    t.quarter AS Quarter,
    SUM(f.sales_amount) AS TotalSales
FROM fact_sales f
JOIN dim_customer c ON f.customer_id = c.customer_src
JOIN dim_time t ON f.time_id = t.rowid
GROUP BY c.ship_country, t.year, t.quarter
ORDER BY c.ship_country, t.year, t.quarter;

#DRILL-DOWN QUERY: Sales by Month for Selected Country
SELECT 
    t.year AS Year,
    t.month AS Month,
    SUM(f.sales_amount) AS TotalSales
FROM fact_sales f
JOIN dim_customer c ON f.customer_id = c.customer_src
JOIN dim_time t ON f.time_id = t.rowid
WHERE c.ship_country = 'Kenya'
GROUP BY t.year, t.month
ORDER BY t.year, t.month;

#SLICE QUERY: Electronics Category
SELECT 
    cat.category_name AS Category,
    SUM(f.sales_amount) AS TotalSales
FROM fact_sales f
JOIN dim_category cat ON f.category_id = cat.category_id
WHERE cat.category_name = 'Electronics'
GROUP BY cat.category_name;



