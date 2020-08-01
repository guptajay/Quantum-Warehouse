# REST API

## Swagger UI
> Explore the API with the help of Swagger UI.

**URL Endpoint:** `/docs`  


***


## Insert / Withdraw Packages
> Insert and Withdraw packages in the Warehouse

**URL Path:** `/package` 

### Insert a Package
> Insert a package in the Warehouse.

**URL Endpoint:** `/insert/{id}` 

**HTTP Method:** POST

#### Request Parameters

| Parameter | Data Type | Required |          Description          |
|:---------:|:---------:|:--------:|:-----------------------------:|
|     id    |  integer  |    Yes   | Specify the ID of the Package |


**Request Example**

`<ROOT_URL>/package/insert/1`

#### Response

| JSON Field |               Description              |
|:----------:|:--------------------------------------:|
|  packageID |       ID of the inserted package       |
| insertPos  | Position where the package is inserted |

The response is given as a status (200).

**Response Example**

200 “Package Inserted”

```
{
  "packageID": 7,
  "insertPos": 49
}
```

### Withdraw a Package
> Withdraw a package in the Warehouse.

**URL Endpoint:** `/withdraw/{id}` 

**HTTP Method:** POST

#### Request Parameters

| Parameter | Data Type | Required |          Description          |
|:---------:|:---------:|:--------:|:-----------------------------:|
|     id    |  integer  |    Yes   | Specify the ID of the Package |


**Request Example**

`<ROOT_URL>/package/withdraw/1`

#### Response

| JSON Field |               Description              |
|:----------:|:--------------------------------------:|
|  packageID |       ID of the inserted package       |
| withrawPos  | Position where the package is withdrawn |

The response is given as a status (200).

**Response Example**

200 “Package Withdrawn

```
{
  "packageID": 7,
  "withdrawPos": 49
}
```

***


## Reset Warehouse
> Reset the Warehouse Environment

**URL Path:** `/reset` 

### Reset

**URL Endpoint:** `/warehouse` 

**HTTP Method:** DELETE

#### Request Parameters

No request parameters


**Request Example**

`<ROOT_URL>/reset/warehouse`

#### Response

| JSON Field |               Description              |
|:----------:|:--------------------------------------:|
|  status    |       Status of the Warehouse          |

The response is given as a status (200).

**Response Example**

200 “Warehouse Successfully Reset”

```
{
  "status": "Warehouse has been successfully reset"
}
```